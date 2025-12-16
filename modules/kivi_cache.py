"""
KIVI-style quantization for low-rank KV cache.

KIVI quantization scheme:
- Key cache: per-channel quantization (quantize along the token dimension)
- Value cache: per-token quantization (quantize along the hidden dimension)

Adapted for low-rank decomposed KV cache where:
- Key latent: [batch, seq_len, rank] -> quantize per-channel (along seq_len)
- Value latent: [batch, seq_len, rank] -> quantize per-token (along rank)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class KIVIQuantizer(nn.Module):
    """
    KIVI-style asymmetric quantizer with per-group min-max scaling.
    
    Args:
        n_bits: Number of quantization bits (2, 3, 4, 8)
        group_size: Size of quantization groups (0 means no grouping)
        residual_length: Number of recent tokens to keep in full precision
        per_channel: If True, quantize per-channel (for Key); else per-token (for Value)
    """
    
    def __init__(
        self,
        n_bits: int = 2,
        group_size: int = 128,
        residual_length: int = 32,
        per_channel: bool = False,
        symmetric: bool = False,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.per_channel = per_channel
        self.symmetric = symmetric
        
        # Quantization range
        if symmetric:
            self.q_max = 2 ** (n_bits - 1) - 1
            self.q_min = -(2 ** (n_bits - 1))
        else:
            self.q_max = 2 ** n_bits - 1
            self.q_min = 0
    
    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor and return quantized values with scale and zero-point.
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            
        Returns:
            quantized: Quantized tensor (int or fake-quantized float)
            scale: Quantization scale
            zero_point: Zero point (for asymmetric quantization)
        """
        if self.n_bits >= 16:
            return x, None, None
        
        original_shape = x.shape
        
        if self.per_channel:
            # Per-channel quantization (for Key): quantize along seq_len dimension
            # Reshape to [batch, dim, seq_len] for channel-wise quantization
            x = x.transpose(-1, -2)  # [batch, dim, seq_len]
            
        # Apply grouping if specified
        if self.group_size > 0 and x.shape[-1] >= self.group_size:
            # Reshape for group-wise quantization
            *leading_dims, last_dim = x.shape
            n_groups = (last_dim + self.group_size - 1) // self.group_size
            padded_dim = n_groups * self.group_size
            
            if padded_dim != last_dim:
                # Pad to make divisible by group_size
                padding = padded_dim - last_dim
                x = F.pad(x, (0, padding), value=0)
            
            x = x.view(*leading_dims, n_groups, self.group_size)
            
            # Compute min/max per group
            if self.symmetric:
                x_absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
                scale = x_absmax / self.q_max
                zero_point = torch.zeros_like(scale)
            else:
                x_min = x.amin(dim=-1, keepdim=True)
                x_max = x.amax(dim=-1, keepdim=True)
                scale = (x_max - x_min).clamp(min=1e-5) / self.q_max
                zero_point = (-x_min / scale).round().clamp(self.q_min, self.q_max)
            
            # Quantize
            x_quant = (x / scale + zero_point).round().clamp(self.q_min, self.q_max)
            
            # Dequantize (fake quantization for training compatibility)
            x_dequant = (x_quant - zero_point) * scale
            
            # Reshape back
            x_dequant = x_dequant.view(*leading_dims, padded_dim)
            if padded_dim != last_dim:
                x_dequant = x_dequant[..., :last_dim]
            
            scale = scale.view(*leading_dims, n_groups)
            zero_point = zero_point.view(*leading_dims, n_groups)
        else:
            # No grouping: quantize entire last dimension
            if self.symmetric:
                x_absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
                scale = x_absmax / self.q_max
                zero_point = torch.zeros_like(scale)
            else:
                x_min = x.amin(dim=-1, keepdim=True)
                x_max = x.amax(dim=-1, keepdim=True)
                scale = (x_max - x_min).clamp(min=1e-5) / self.q_max
                zero_point = (-x_min / scale).round().clamp(self.q_min, self.q_max)
            
            x_quant = (x / scale + zero_point).round().clamp(self.q_min, self.q_max)
            x_dequant = (x_quant - zero_point) * scale
        
        if self.per_channel:
            # Transpose back for Key
            x_dequant = x_dequant.transpose(-1, -2)
        
        return x_dequant, scale, zero_point
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake quantization: quantize and immediately dequantize."""
        x_dequant, _, _ = self.quantize(x)
        return x_dequant


class KIVICache(nn.Module):
    """
    KIVI-style quantized cache for low-rank KV.
    
    Maintains:
    - Quantized cache for older tokens
    - Full-precision cache for recent tokens (residual)
    
    Key insight: In low-rank setting, we quantize the latent representations
    (output of BLinear) rather than the full KV states.
    """
    
    def __init__(
        self,
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 128,
        residual_length: int = 32,
    ):
        super().__init__()
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length
        
        # Key quantizer: per-channel (along token dimension)
        self.key_quantizer = KIVIQuantizer(
            n_bits=k_bits,
            group_size=group_size,
            residual_length=residual_length,
            per_channel=True,  # Key uses per-channel
            symmetric=False,
        )
        
        # Value quantizer: per-token (along hidden dimension)  
        self.value_quantizer = KIVIQuantizer(
            n_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
            per_channel=False,  # Value uses per-token
            symmetric=False,
        )
    
    @torch.no_grad()
    def quantize_key(self, key_latent: torch.Tensor) -> torch.Tensor:
        """
        Quantize key latent using per-channel quantization.
        
        Args:
            key_latent: [batch, seq_len, rank] - output of k_proj.BLinear
        """
        return self.key_quantizer(key_latent)
    
    @torch.no_grad()
    def quantize_value(self, value_latent: torch.Tensor) -> torch.Tensor:
        """
        Quantize value latent using per-token quantization.
        
        Args:
            value_latent: [batch, seq_len, rank] - output of v_proj.BLinear
        """
        return self.value_quantizer(value_latent)


class KIVIDynamicCache:
    """
    Dynamic cache that stores quantized historical tokens and full-precision recent tokens.
    
    This follows KIVI's design where:
    - Historical tokens are quantized to save memory
    - Recent tokens (residual_length) are kept in full precision for accuracy
    """
    
    def __init__(
        self,
        num_layers: int,
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 128,
        residual_length: int = 32,
    ):
        self.num_layers = num_layers
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length
        
        # Per-layer cache
        # Each entry: (key_quant, key_full, key_scale, key_zp, 
        #              value_quant, value_full, value_scale, value_zp)
        self.cache: List[Optional[Tuple]] = [None] * num_layers
        self._seen_tokens = 0
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length in the cache."""
        if self.cache[layer_idx] is None:
            return 0
        
        key_quant, key_full, _, _, _, _, _, _ = self.cache[layer_idx]
        quant_len = key_quant.shape[1] if key_quant is not None else 0
        full_len = key_full.shape[1] if key_full is not None else 0
        return quant_len + full_len
    
    @torch.no_grad()
    def update(
        self,
        key_latent: torch.Tensor,
        value_latent: torch.Tensor,
        layer_idx: int,
        key_quantizer: KIVIQuantizer,
        value_quantizer: KIVIQuantizer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value latents.
        
        During prefill: accumulate tokens, quantize when exceeding residual_length
        During decode: append to full-precision cache, move to quantized when needed
        
        Args:
            key_latent: [batch, new_seq_len, rank]
            value_latent: [batch, new_seq_len, rank]
            layer_idx: Layer index
            key_quantizer: Quantizer for keys
            value_quantizer: Quantizer for values
            
        Returns:
            full_key_latent: All key latents (quantized + full precision, dequantized)
            full_value_latent: All value latents (quantized + full precision, dequantized)
        """
        batch_size, new_seq_len, rank = key_latent.shape
        
        if self.cache[layer_idx] is None:
            # First call: initialize cache
            if new_seq_len <= self.residual_length:
                # All tokens fit in residual
                self.cache[layer_idx] = (
                    None, key_latent, None, None,  # key: no quant, all full
                    None, value_latent, None, None  # value: no quant, all full
                )
                return key_latent, value_latent
            else:
                # Need to quantize some tokens
                n_quant = new_seq_len - self.residual_length
                # Ensure quantized portion is aligned to group_size
                if self.group_size > 0:
                    n_quant = (n_quant // self.group_size) * self.group_size
                
                if n_quant > 0:
                    key_to_quant = key_latent[:, :n_quant, :]
                    key_full = key_latent[:, n_quant:, :]
                    value_to_quant = value_latent[:, :n_quant, :]
                    value_full = value_latent[:, n_quant:, :]
                    
                    key_quant, key_scale, key_zp = key_quantizer.quantize(key_to_quant)
                    value_quant, value_scale, value_zp = value_quantizer.quantize(value_to_quant)
                    
                    self.cache[layer_idx] = (
                        key_quant, key_full, key_scale, key_zp,
                        value_quant, value_full, value_scale, value_zp
                    )
                    
                    all_keys = torch.cat([key_quant, key_full], dim=1)
                    all_values = torch.cat([value_quant, value_full], dim=1)
                    return all_keys, all_values
                else:
                    self.cache[layer_idx] = (
                        None, key_latent, None, None,
                        None, value_latent, None, None
                    )
                    return key_latent, value_latent
        else:
            # Append to existing cache
            key_quant, key_full, key_scale, key_zp, \
            value_quant, value_full, value_scale, value_zp = self.cache[layer_idx]
            
            # Append new tokens to full-precision cache
            if key_full is not None:
                key_full = torch.cat([key_full, key_latent], dim=1)
                value_full = torch.cat([value_full, value_latent], dim=1)
            else:
                key_full = key_latent
                value_full = value_latent
            
            # Check if we need to move tokens to quantized cache
            full_len = key_full.shape[1]
            
            if full_len > self.residual_length:
                n_to_move = full_len - self.residual_length
                # Align to group_size
                if self.group_size > 0:
                    n_to_move = (n_to_move // self.group_size) * self.group_size
                
                if n_to_move > 0:
                    # Quantize and move
                    key_to_move = key_full[:, :n_to_move, :]
                    value_to_move = value_full[:, :n_to_move, :]
                    
                    key_move_quant, key_move_scale, key_move_zp = key_quantizer.quantize(key_to_move)
                    value_move_quant, value_move_scale, value_move_zp = value_quantizer.quantize(value_to_move)
                    
                    # Merge with existing quantized cache
                    if key_quant is not None:
                        key_quant = torch.cat([key_quant, key_move_quant], dim=1)
                        value_quant = torch.cat([value_quant, value_move_quant], dim=1)
                        # Note: scales need proper merging for grouped quantization
                        # For simplicity, we store dequantized values here
                    else:
                        key_quant = key_move_quant
                        value_quant = value_move_quant
                    
                    key_full = key_full[:, n_to_move:, :]
                    value_full = value_full[:, n_to_move:, :]
            
            # Update cache
            self.cache[layer_idx] = (
                key_quant, key_full, key_scale, key_zp,
                value_quant, value_full, value_scale, value_zp
            )
            
            # Return concatenated (all dequantized for downstream use)
            if key_quant is not None:
                all_keys = torch.cat([key_quant, key_full], dim=1)
                all_values = torch.cat([value_quant, value_full], dim=1)
            else:
                all_keys = key_full
                all_values = value_full
            
            return all_keys, all_values


def create_kivi_quantizers(
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 128,
    residual_length: int = 32,
) -> Tuple[KIVIQuantizer, KIVIQuantizer]:
    """
    Create KIVI-style quantizers for Key and Value.
    
    Args:
        k_bits: Bits for Key quantization
        v_bits: Bits for Value quantization
        group_size: Quantization group size
        residual_length: Number of recent tokens to keep in full precision
        
    Returns:
        key_quantizer: Per-channel quantizer for Key
        value_quantizer: Per-token quantizer for Value
    """
    key_quantizer = KIVIQuantizer(
        n_bits=k_bits,
        group_size=group_size,
        residual_length=residual_length,
        per_channel=True,
        symmetric=False,
    )
    
    value_quantizer = KIVIQuantizer(
        n_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
        per_channel=False,
        symmetric=False,
    )
    
    return key_quantizer, value_quantizer
