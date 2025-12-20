"""
General KIVI-style Quantized KV Cache for Standard Transformer Models (Qwen, Llama, etc.)

This module provides a drop-in replacement for DynamicCache that works with
standard transformer attention (4D key/value tensors: [batch, heads, seq, head_dim]).

Key features:
- Per-channel quantization for Keys (along token dimension)
- Per-token quantization for Values (along head_dim dimension)  
- Recent tokens kept in full precision (residual)
- Compatible with model.generate()

Usage:
    from modules.kivi_cache_general import KIVICache, apply_kivi_to_model
    
    # Method 1: Use cache directly
    cache = KIVICache(k_bits=2, v_bits=2)
    outputs = model.generate(input_ids, past_key_values=cache, use_cache=True)
    
    # Method 2: Patch model to auto-create cache
    apply_kivi_to_model(model, k_bits=2, v_bits=2)
    outputs = model.generate(input_ids, use_cache=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Any, Dict, Union
from transformers.cache_utils import Cache, DynamicCache
import math
import warnings


class KIVIQuantizer(nn.Module):
    """
    KIVI-style asymmetric quantizer with per-group min-max scaling.
    
    For Key: per_channel=True (quantize along token dimension)
    For Value: per_channel=False (quantize along head_dim dimension)
    """
    
    def __init__(
        self,
        n_bits: int = 2,
        group_size: int = 32,
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
        """
        Quantize tensor and return dequantized values with scale/zero_point.
        
        Args:
            x: Input tensor of shape [batch, heads, seq_len, head_dim]
            
        Returns:
            x_dequant: Fake-quantized tensor (same shape)
            scale: Quantization scales
            zero_point: Zero points
        """
        if self.n_bits >= 16:
            return x, None, None
        
        # For per-channel (Key): quantize along seq_len dimension
        # For per-token (Value): quantize along head_dim dimension
        if self.per_channel:
            # [B, H, S, D] -> [B, H, D, S] for per-channel quantization
            x = x.transpose(-1, -2)
        
        # Shape: [B, H, ..., quant_dim]
        *leading_dims, quant_dim = x.shape
        
        if self.group_size > 0 and quant_dim >= self.group_size:
            n_groups = quant_dim // self.group_size
            actual_dim = n_groups * self.group_size
            
            # Only quantize aligned part
            if actual_dim < quant_dim:
                x_to_quant = x[..., :actual_dim]
                x_residual = x[..., actual_dim:]
            else:
                x_to_quant = x
                x_residual = None
            
            # Reshape for group quantization
            x_grouped = x_to_quant.view(*leading_dims, n_groups, self.group_size)
            
            # Min-max per group
            x_min = x_grouped.amin(dim=-1, keepdim=True)
            x_max = x_grouped.amax(dim=-1, keepdim=True)
            scale = (x_max - x_min).clamp(min=1e-5) / self.q_max
            # zero_point should NOT be clamped - it can be any value
            # zero_point = -x_min / scale (mathematically equivalent to using x_min directly)
            zero_point = -x_min / scale
            
            # Quantize and dequantize
            x_quant = (x_grouped / scale + zero_point).round().clamp(self.q_min, self.q_max)
            x_dequant = (x_quant - zero_point) * scale
            
            # Reshape back
            x_dequant = x_dequant.view(*leading_dims, actual_dim)
            
            if x_residual is not None:
                x_dequant = torch.cat([x_dequant, x_residual], dim=-1)
        else:
            # Quantize entire dimension
            x_min = x.amin(dim=-1, keepdim=True)
            x_max = x.amax(dim=-1, keepdim=True)
            scale = (x_max - x_min).clamp(min=1e-5) / self.q_max
            # zero_point should NOT be clamped
            zero_point = -x_min / scale
            
            x_quant = (x / scale + zero_point).round().clamp(self.q_min, self.q_max)
            x_dequant = (x_quant - zero_point) * scale
        
        # Transpose back for per-channel
        if self.per_channel:
            x_dequant = x_dequant.transpose(-1, -2)
        
        return x_dequant, scale, zero_point
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dequant, _, _ = self.quantize(x)
        return x_dequant


class KIVICache(Cache):
    """
    KIVI-style Quantized KV Cache for Standard Transformer Models.
    
    This cache works with standard 4D key/value tensors:
    - Shape: [batch_size, num_heads, seq_len, head_dim]
    
    Quantization strategy (following KIVI paper):
    - Key: per-channel quantization (along token/seq dimension)
    - Value: per-token quantization (along head_dim dimension)
    - Recent tokens: kept in full precision for accuracy
    
    Compatible with:
    - Qwen, Qwen2, Qwen2.5
    - Llama, Llama2, Llama3
    - Any model using standard transformers cache interface
    
    Example:
        cache = KIVICache(k_bits=2, v_bits=2, residual_length=128)
        outputs = model.generate(
            input_ids,
            past_key_values=cache,
            use_cache=True,
            max_new_tokens=100,
        )
    """
    
    def __init__(
        self,
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 32,
        residual_length: int = 128,
    ):
        """
        Initialize KIVI Cache.
        
        Args:
            k_bits: Bits for Key quantization (2, 4, 8, or 16 for no quantization)
            v_bits: Bits for Value quantization (2, 4, 8, or 16 for no quantization)
            group_size: Group size for quantization (smaller = better accuracy, more overhead)
            residual_length: Number of recent tokens to keep in full precision
        """
        super().__init__()
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length
        
        # Per-layer storage
        # Format: {layer_idx: (key_quant, key_residual, value_quant, value_residual)}
        self._cache: Dict[int, Tuple[
            Optional[torch.Tensor],  # key_quant
            Optional[torch.Tensor],  # key_residual
            Optional[torch.Tensor],  # value_quant
            Optional[torch.Tensor],  # value_residual
        ]] = {}
        
        # Quantizers
        self.key_quantizer = KIVIQuantizer(n_bits=k_bits, group_size=group_size, per_channel=True)
        self.value_quantizer = KIVIQuantizer(n_bits=v_bits, group_size=group_size, per_channel=False)
        
        # Track seen tokens
        self._seen_tokens = 0
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __iter__(self):
        for layer_idx in sorted(self._cache.keys()):
            yield self[layer_idx]
    
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get combined key/value for a layer."""
        if layer_idx not in self._cache:
            raise KeyError(f"Layer {layer_idx} not in cache")
        
        key_quant, key_residual, value_quant, value_residual = self._cache[layer_idx]
        
        # Combine quantized and residual parts
        if key_quant is not None and key_residual is not None:
            all_keys = torch.cat([key_quant, key_residual], dim=2)  # dim=2 is seq_len
            all_values = torch.cat([value_quant, value_residual], dim=2)
        elif key_residual is not None:
            all_keys = key_residual
            all_values = value_residual
        else:
            all_keys = key_quant
            all_values = value_quant
        
        return all_keys, all_values
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length in cache."""
        if layer_idx not in self._cache:
            return 0
        
        key_quant, key_residual, _, _ = self._cache[layer_idx]
        quant_len = key_quant.shape[2] if key_quant is not None else 0
        residual_len = key_residual.shape[2] if key_residual is not None else 0
        return quant_len + residual_len
    
    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)
    
    def get_max_length(self) -> Optional[int]:
        return None
    
    @property
    def seen_tokens(self) -> int:
        if not self._cache:
            return 0
        return self.get_seq_length(0)
    
    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states.
        
        Args:
            key_states: [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim]
            layer_idx: Current layer index
            cache_kwargs: Additional kwargs (unused, for compatibility)
            
        Returns:
            Updated key and value states (full sequence)
        """
        batch_size, num_heads, new_seq_len, head_dim = key_states.shape
        
        if layer_idx not in self._cache:
            # First call for this layer
            if layer_idx == 0:
                self._seen_tokens = new_seq_len
            
            if new_seq_len <= self.residual_length:
                # All tokens fit in residual
                self._cache[layer_idx] = (None, key_states.clone(), None, value_states.clone())
                return key_states, value_states
            else:
                # Need to quantize some tokens
                n_quant = new_seq_len - self.residual_length
                # Align to group_size
                if self.group_size > 0:
                    n_quant = (n_quant // self.group_size) * self.group_size
                
                if n_quant > 0:
                    # Split and quantize
                    key_to_quant = key_states[:, :, :n_quant, :]
                    key_residual = key_states[:, :, n_quant:, :].clone()
                    value_to_quant = value_states[:, :, :n_quant, :]
                    value_residual = value_states[:, :, n_quant:, :].clone()
                    
                    key_quant = self.key_quantizer(key_to_quant)
                    value_quant = self.value_quantizer(value_to_quant)
                    
                    self._cache[layer_idx] = (key_quant, key_residual, value_quant, value_residual)
                    
                    all_keys = torch.cat([key_quant, key_residual], dim=2)
                    all_values = torch.cat([value_quant, value_residual], dim=2)
                    return all_keys, all_values
                else:
                    self._cache[layer_idx] = (None, key_states.clone(), None, value_states.clone())
                    return key_states, value_states
        else:
            # Append to existing cache
            key_quant, key_residual, value_quant, value_residual = self._cache[layer_idx]
            
            if layer_idx == 0:
                self._seen_tokens += new_seq_len
            
            # Append new tokens to residual
            if key_residual is not None:
                key_residual = torch.cat([key_residual, key_states], dim=2)
                value_residual = torch.cat([value_residual, value_states], dim=2)
            else:
                key_residual = key_states.clone()
                value_residual = value_states.clone()
            
            residual_len = key_residual.shape[2]
            
            # Check if we need to move tokens to quantized cache
            if residual_len > self.residual_length:
                n_to_move = residual_len - self.residual_length
                if self.group_size > 0:
                    n_to_move = (n_to_move // self.group_size) * self.group_size
                
                if n_to_move > 0:
                    # Quantize and move oldest tokens
                    key_to_move = key_residual[:, :, :n_to_move, :]
                    value_to_move = value_residual[:, :, :n_to_move, :]
                    
                    key_move_quant = self.key_quantizer(key_to_move)
                    value_move_quant = self.value_quantizer(value_to_move)
                    
                    if key_quant is not None:
                        key_quant = torch.cat([key_quant, key_move_quant], dim=2)
                        value_quant = torch.cat([value_quant, value_move_quant], dim=2)
                    else:
                        key_quant = key_move_quant
                        value_quant = value_move_quant
                    
                    key_residual = key_residual[:, :, n_to_move:, :]
                    value_residual = value_residual[:, :, n_to_move:, :]
            
            self._cache[layer_idx] = (key_quant, key_residual, value_quant, value_residual)
            
            # Return full sequence
            if key_quant is not None:
                all_keys = torch.cat([key_quant, key_residual], dim=2)
                all_values = torch.cat([value_quant, value_residual], dim=2)
            else:
                all_keys = key_residual
                all_values = value_residual
            
            return all_keys, all_values
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search."""
        for layer_idx in self._cache:
            key_quant, key_residual, value_quant, value_residual = self._cache[layer_idx]
            
            if key_quant is not None:
                key_quant = key_quant.index_select(0, beam_idx)
                value_quant = value_quant.index_select(0, beam_idx)
            if key_residual is not None:
                key_residual = key_residual.index_select(0, beam_idx)
                value_residual = value_residual.index_select(0, beam_idx)
            
            self._cache[layer_idx] = (key_quant, key_residual, value_quant, value_residual)
    
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to legacy tuple format."""
        legacy = []
        for layer_idx in sorted(self._cache.keys()):
            keys, values = self[layer_idx]
            legacy.append((keys, values))
        return tuple(legacy)
    
    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 32,
        residual_length: int = 128,
    ) -> "KIVICache":
        """Create KIVICache from legacy cache format."""
        cache = cls(k_bits=k_bits, v_bits=v_bits, group_size=group_size, residual_length=residual_length)
        for layer_idx, (key_states, value_states) in enumerate(past_key_values):
            cache.update(key_states, value_states, layer_idx)
        return cache
    
    def reset(self):
        """Clear all cached data."""
        self._cache.clear()
        self._seen_tokens = 0
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._cache:
            return {"status": "empty"}
        
        layer_0 = self._cache.get(0)
        if layer_0 is None:
            return {"status": "no layer 0"}
        
        key_quant, key_residual, _, _ = layer_0
        
        quant_len = key_quant.shape[2] if key_quant is not None else 0
        residual_len = key_residual.shape[2] if key_residual is not None else 0
        
        # Calculate memory savings
        total_tokens = quant_len + residual_len
        if total_tokens > 0:
            effective_bits = (
                quant_len * (self.k_bits + self.v_bits) / 2 + 
                residual_len * 16
            ) / total_tokens
        else:
            effective_bits = 16
        
        return {
            "num_layers": len(self._cache),
            "total_seq_len": total_tokens,
            "quantized_len": quant_len,
            "residual_len": residual_len,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "group_size": self.group_size,
            "residual_length": self.residual_length,
            "effective_bits": f"{effective_bits:.2f}",
            "memory_ratio": f"{effective_bits / 16:.2%}",
        }


def create_kivi_cache(
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
) -> KIVICache:
    """
    Factory function to create KIVI cache.
    
    Args:
        k_bits: Key quantization bits (2, 4, 8, 16)
        v_bits: Value quantization bits (2, 4, 8, 16)
        group_size: Quantization group size
        residual_length: Full precision residual length
        
    Returns:
        KIVICache instance
    """
    return KIVICache(
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
    )


# ============================================================================
# Model Patching Utilities for Qwen
# ============================================================================

def _patch_qwen2_attention(
    attn_module,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
):
    """
    Patch a Qwen2 attention module to support KIVICache.
    
    This modifies the attention's forward to detect and use KIVICache.
    """
    original_forward = attn_module.forward
    
    def patched_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        # If using KIVICache, it will be handled by the standard cache.update() interface
        # No special handling needed as KIVICache is compatible with Cache interface
        return original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
    
    attn_module.forward = patched_forward


def apply_kivi_to_model(
    model,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
):
    """
    Apply KIVI configuration to a model.
    
    This stores KIVI parameters on the model for later cache creation.
    
    Args:
        model: The transformer model (Qwen, Llama, etc.)
        k_bits: Key quantization bits
        v_bits: Value quantization bits
        group_size: Quantization group size
        residual_length: Full precision residual length
    
    Usage:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")
        apply_kivi_to_model(model, k_bits=2, v_bits=2)
        
        # Then use with generate:
        cache = KIVICache(k_bits=2, v_bits=2)
        outputs = model.generate(input_ids, past_key_values=cache)
    """
    model.kivi_config = {
        "k_bits": k_bits,
        "v_bits": v_bits,
        "group_size": group_size,
        "residual_length": residual_length,
    }
    
    # Store reference for easy cache creation
    model.create_kivi_cache = lambda: KIVICache(
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
    )
    
    print(f"KIVI config applied: k_bits={k_bits}, v_bits={v_bits}, "
          f"group_size={group_size}, residual_length={residual_length}")
    print("Use: cache = model.create_kivi_cache(); model.generate(..., past_key_values=cache)")


# ============================================================================
# Monkey-patch model.generate() to auto-create KIVI cache
# ============================================================================

def patch_model_generate(
    model,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
    auto_reset: bool = True,
):
    """
    Patch model.generate() to automatically use KIVI cache.
    
    Args:
        model: The model to patch
        k_bits, v_bits, group_size, residual_length: KIVI parameters
        auto_reset: Whether to reset cache between generate calls
        
    Usage:
        patch_model_generate(model, k_bits=2, v_bits=2)
        outputs = model.generate(input_ids)  # Automatically uses KIVI cache
    """
    original_generate = model.generate
    
    # Create persistent cache
    kivi_cache = KIVICache(
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
    )
    
    def patched_generate(
        inputs=None,
        generation_config=None,
        **kwargs,
    ):
        nonlocal kivi_cache
        
        # Reset cache for new generation
        if auto_reset:
            kivi_cache.reset()
        
        # Use KIVI cache unless user provides their own
        if 'past_key_values' not in kwargs or kwargs['past_key_values'] is None:
            kwargs['past_key_values'] = kivi_cache
        
        # Ensure use_cache is True
        kwargs['use_cache'] = True
        
        return original_generate(inputs, generation_config=generation_config, **kwargs)
    
    model.generate = patched_generate
    model._kivi_cache = kivi_cache
    
    print(f"Model.generate() patched with KIVI: k={k_bits}bit, v={v_bits}bit")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "KIVICache",
    "KIVIQuantizer",
    "create_kivi_cache",
    "apply_kivi_to_model",
    "patch_model_generate",
]
