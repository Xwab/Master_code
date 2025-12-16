"""
KIVI-style quantized cache for low-rank KV cache.

This module provides a drop-in replacement for transformers' DynamicCache
that stores low-rank KV latents with KIVI-style quantization:
- Key: per-channel quantization (along token dimension)
- Value: per-token quantization (along hidden dimension)
- Recent tokens kept in full precision (residual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Any, Dict
import math


class KIVIQuantizer(nn.Module):
    """
    KIVI-style asymmetric quantizer with per-group min-max scaling.
    """
    
    def __init__(
        self,
        n_bits: int = 2,
        group_size: int = 128,
        per_channel: bool = False,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.per_channel = per_channel
        self.q_max = 2 ** n_bits - 1
        self.q_min = 0
    
    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize and return quantized values with scale and zero-point."""
        if self.n_bits >= 16:
            return x, None, None
        
        if self.per_channel:
            x = x.transpose(-1, -2)
        
        # Group-wise quantization
        if self.group_size > 0 and x.shape[-1] >= self.group_size:
            *leading_dims, last_dim = x.shape
            n_groups = (last_dim + self.group_size - 1) // self.group_size
            padded_dim = n_groups * self.group_size
            
            if padded_dim != last_dim:
                padding = padded_dim - last_dim
                x = F.pad(x, (0, padding), value=0)
            
            x = x.view(*leading_dims, n_groups, self.group_size)
            
            x_min = x.amin(dim=-1, keepdim=True)
            x_max = x.amax(dim=-1, keepdim=True)
            scale = (x_max - x_min).clamp(min=1e-5) / self.q_max
            zero_point = (-x_min / scale).round().clamp(self.q_min, self.q_max)
            
            x_quant = (x / scale + zero_point).round().clamp(self.q_min, self.q_max)
            x_dequant = (x_quant - zero_point) * scale
            
            x_dequant = x_dequant.view(*leading_dims, padded_dim)
            if padded_dim != last_dim:
                x_dequant = x_dequant[..., :last_dim]
            
            scale = scale.view(*leading_dims, n_groups)
            zero_point = zero_point.view(*leading_dims, n_groups)
        else:
            x_min = x.amin(dim=-1, keepdim=True)
            x_max = x.amax(dim=-1, keepdim=True)
            scale = (x_max - x_min).clamp(min=1e-5) / self.q_max
            zero_point = (-x_min / scale).round().clamp(self.q_min, self.q_max)
            
            x_quant = (x / scale + zero_point).round().clamp(self.q_min, self.q_max)
            x_dequant = (x_quant - zero_point) * scale
        
        if self.per_channel:
            x_dequant = x_dequant.transpose(-1, -2)
        
        return x_dequant, scale, zero_point
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dequant, _, _ = self.quantize(x)
        return x_dequant


class KIVILatentCache:
    """
    KIVI-style cache for low-rank KV latents.
    
    This cache stores the output of BLinear (low-rank projection) with KIVI quantization.
    It maintains:
    - Quantized cache for older tokens (memory efficient)
    - Full-precision cache for recent tokens (accuracy)
    
    Compatible with the transformers Cache interface.
    """
    
    def __init__(
        self,
        k_bits: int = 2,
        v_bits: int = 2, 
        group_size: int = 128,
        residual_length: int = 32,
    ):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length
        
        # Per-layer storage
        # Format: {layer_idx: (key_quant, key_residual, value_quant, value_residual)}
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        
        # Quantizers
        self.key_quantizer = KIVIQuantizer(n_bits=k_bits, group_size=group_size, per_channel=True)
        self.value_quantizer = KIVIQuantizer(n_bits=v_bits, group_size=group_size, per_channel=False)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length in cache."""
        if layer_idx not in self._cache:
            return 0
        
        key_quant, key_residual, _, _ = self._cache[layer_idx]
        quant_len = key_quant.shape[1] if key_quant is not None else 0
        residual_len = key_residual.shape[1] if key_residual is not None else 0
        return quant_len + residual_len
    
    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Get the usable length considering new tokens."""
        return self.get_seq_length(layer_idx)
    
    @torch.no_grad()
    def update(
        self,
        key_latent: torch.Tensor,
        value_latent: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value latents.
        
        Args:
            key_latent: [batch, seq_len, rank] - output of k_proj.BLinear
            value_latent: [batch, seq_len, rank] - output of v_proj.BLinear
            layer_idx: Current layer index
            
        Returns:
            all_keys: Concatenated key latents (quantized + residual)
            all_values: Concatenated value latents (quantized + residual)
        """
        batch_size, new_seq_len, rank = key_latent.shape
        
        if layer_idx not in self._cache:
            # First call: initialize cache
            if new_seq_len <= self.residual_length:
                # All tokens fit in residual (no quantization)
                self._cache[layer_idx] = (None, key_latent, None, value_latent)
                return key_latent, value_latent
            else:
                # Need to quantize some tokens
                n_quant = new_seq_len - self.residual_length
                if self.group_size > 0:
                    n_quant = (n_quant // self.group_size) * self.group_size
                
                if n_quant > 0:
                    # Quantize older tokens
                    key_to_quant = key_latent[:, :n_quant, :]
                    key_residual = key_latent[:, n_quant:, :]
                    value_to_quant = value_latent[:, :n_quant, :]
                    value_residual = value_latent[:, n_quant:, :]
                    
                    key_quant = self.key_quantizer(key_to_quant)
                    value_quant = self.value_quantizer(value_to_quant)
                    
                    self._cache[layer_idx] = (key_quant, key_residual, value_quant, value_residual)
                    
                    return torch.cat([key_quant, key_residual], dim=1), \
                           torch.cat([value_quant, value_residual], dim=1)
                else:
                    self._cache[layer_idx] = (None, key_latent, None, value_latent)
                    return key_latent, value_latent
        else:
            # Append to existing cache
            key_quant, key_residual, value_quant, value_residual = self._cache[layer_idx]
            
            # Append new tokens to residual
            if key_residual is not None:
                key_residual = torch.cat([key_residual, key_latent], dim=1)
                value_residual = torch.cat([value_residual, value_latent], dim=1)
            else:
                key_residual = key_latent
                value_residual = value_latent
            
            # Check if we need to move tokens to quantized cache
            residual_len = key_residual.shape[1]
            
            if residual_len > self.residual_length:
                n_to_move = residual_len - self.residual_length
                if self.group_size > 0:
                    n_to_move = (n_to_move // self.group_size) * self.group_size
                
                if n_to_move > 0:
                    # Quantize and move oldest tokens from residual
                    key_to_move = key_residual[:, :n_to_move, :]
                    value_to_move = value_residual[:, :n_to_move, :]
                    
                    key_move_quant = self.key_quantizer(key_to_move)
                    value_move_quant = self.value_quantizer(value_to_move)
                    
                    # Merge with existing quantized cache
                    if key_quant is not None:
                        key_quant = torch.cat([key_quant, key_move_quant], dim=1)
                        value_quant = torch.cat([value_quant, value_move_quant], dim=1)
                    else:
                        key_quant = key_move_quant
                        value_quant = value_move_quant
                    
                    # Update residual
                    key_residual = key_residual[:, n_to_move:, :]
                    value_residual = value_residual[:, n_to_move:, :]
            
            # Update cache
            self._cache[layer_idx] = (key_quant, key_residual, value_quant, value_residual)
            
            # Return concatenated (dequantized + residual)
            if key_quant is not None:
                all_keys = torch.cat([key_quant, key_residual], dim=1)
                all_values = torch.cat([value_quant, value_residual], dim=1)
            else:
                all_keys = key_residual
                all_values = value_residual
            
            return all_keys, all_values
    
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached key/value for a layer (for compatibility)."""
        if layer_idx not in self._cache:
            return None, None
        
        key_quant, key_residual, value_quant, value_residual = self._cache[layer_idx]
        
        if key_quant is not None:
            all_keys = torch.cat([key_quant, key_residual], dim=1)
            all_values = torch.cat([value_quant, value_residual], dim=1)
        else:
            all_keys = key_residual
            all_values = value_residual
        
        return all_keys, all_values
    
    def __len__(self) -> int:
        """Return number of layers cached."""
        return len(self._cache)
    
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to legacy cache format for compatibility."""
        legacy = []
        for layer_idx in sorted(self._cache.keys()):
            keys, values = self[layer_idx]
            legacy.append((keys, values))
        return tuple(legacy)
    
    def reset(self):
        """Clear all cached data."""
        self._cache.clear()
    
    def seen_tokens(self) -> int:
        """Return total number of seen tokens."""
        if not self._cache:
            return 0
        return self.get_seq_length(0)


def create_kivi_cache(
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 128,
    residual_length: int = 32,
) -> KIVILatentCache:
    """
    Factory function to create a KIVI cache.
    
    Args:
        k_bits: Bits for Key quantization (2, 3, 4, 8, 16)
        v_bits: Bits for Value quantization (2, 3, 4, 8, 16)
        group_size: Group size for quantization
        residual_length: Number of recent tokens to keep in full precision
        
    Returns:
        KIVILatentCache instance
    """
    return KIVILatentCache(
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
    )


# Legacy exports for compatibility
KIVICache = KIVILatentCache
KIVIDynamicCache = KIVILatentCache

def create_kivi_quantizers(
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 128,
    residual_length: int = 32,
) -> Tuple[KIVIQuantizer, KIVIQuantizer]:
    """Create KIVI-style quantizers for Key and Value."""
    key_quantizer = KIVIQuantizer(n_bits=k_bits, group_size=group_size, per_channel=True)
    value_quantizer = KIVIQuantizer(n_bits=v_bits, group_size=group_size, per_channel=False)
    return key_quantizer, value_quantizer
