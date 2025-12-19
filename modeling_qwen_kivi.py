"""
Qwen Model with KIVI Quantized KV Cache

This module provides a patched version of Qwen models that uses KIVI-style
KV cache quantization for memory-efficient long-context inference.

Supports:
- Qwen2
- Qwen2.5
- Qwen (original)

Usage:
    from modeling_qwen_kivi import load_qwen_with_kivi, KIVICache
    
    # Load model with KIVI support
    model, tokenizer = load_qwen_with_kivi(
        "Qwen/Qwen2-7B-Instruct",
        k_bits=2, 
        v_bits=2,
    )
    
    # Generate with automatic KIVI cache
    outputs = model.generate(input_ids, max_new_tokens=100)
    
    # Or manually control the cache
    cache = KIVICache(k_bits=2, v_bits=2)
    outputs = model.generate(input_ids, past_key_values=cache, use_cache=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache, DynamicCache
import warnings
import math

# Import our KIVI cache
from modules.kivi_cache_general import KIVICache, KIVIQuantizer, create_kivi_cache


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    """Create causal attention mask."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(tgt_len, device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(tgt_len, 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([
            torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device),
            mask
        ], dim=-1)
    
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class Qwen2KIVIAttention(nn.Module):
    """
    Qwen2 attention with KIVI cache support.
    
    This is a drop-in replacement that adds KIVI quantization
    while maintaining full compatibility with the original attention.
    """
    
    def __init__(self, original_attn, layer_idx: int = 0):
        super().__init__()
        self.original_attn = original_attn
        self.layer_idx = layer_idx
        
        # Copy attributes from original
        self.config = original_attn.config
        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.num_key_value_heads = original_attn.num_key_value_heads
        self.num_key_value_groups = original_attn.num_key_value_groups
        
        # Copy modules
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.rotary_emb = original_attn.rotary_emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward with KIVI cache support."""
        
        bsz, q_len, _ = hidden_states.size()
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if hasattr(self.rotary_emb, 'forward'):
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        
        # Update cache
        if past_key_value is not None:
            # KIVICache and DynamicCache both support this interface
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs=None
            )
        
        # Repeat K/V for GQA
        if self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            causal_mask = attention_mask
            if causal_mask.dim() == 2:
                causal_mask = causal_mask[:, None, None, :]
            elif causal_mask.dim() == 3:
                causal_mask = causal_mask[:, None, :, :]
            attn_weights = attn_weights + causal_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Apply rotary position embeddings."""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for GQA."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def patch_qwen_model(
    model,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
    auto_inject_cache: bool = True,
):
    """
    Patch a Qwen model to use KIVI cache.
    
    Args:
        model: Qwen model (Qwen2ForCausalLM, etc.)
        k_bits: Key quantization bits
        v_bits: Value quantization bits
        group_size: Quantization group size
        residual_length: Full precision residual length
        auto_inject_cache: Whether to auto-inject KIVI cache in generate()
    """
    # Store KIVI config
    model.kivi_config = {
        "k_bits": k_bits,
        "v_bits": v_bits,
        "group_size": group_size,
        "residual_length": residual_length,
    }
    
    # Add cache creation method
    def create_kivi_cache():
        return KIVICache(
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
        )
    model.create_kivi_cache = create_kivi_cache
    
    # Patch generate if requested
    if auto_inject_cache:
        original_generate = model.generate
        
        def patched_generate(
            inputs=None,
            generation_config=None,
            **kwargs,
        ):
            # Create fresh KIVI cache for each generation
            if 'past_key_values' not in kwargs or kwargs['past_key_values'] is None:
                kwargs['past_key_values'] = create_kivi_cache()
            
            # Ensure use_cache is True
            if 'use_cache' not in kwargs:
                kwargs['use_cache'] = True
            
            return original_generate(inputs, generation_config=generation_config, **kwargs)
        
        model.generate = patched_generate
        model._original_generate = original_generate
    
    print(f"Qwen model patched with KIVI: k={k_bits}bit, v={v_bits}bit, "
          f"group_size={group_size}, residual={residual_length}")
    
    return model


def load_qwen_with_kivi(
    model_name_or_path: str,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
    auto_inject_cache: bool = True,
    **model_kwargs,
) -> Tuple[Any, Any]:
    """
    Load a Qwen model with KIVI cache support.
    
    Args:
        model_name_or_path: Model identifier or path
        k_bits: Key quantization bits (2, 4, 8, 16)
        v_bits: Value quantization bits (2, 4, 8, 16)
        group_size: Quantization group size
        residual_length: Full precision residual length
        device_map: Device placement strategy
        torch_dtype: Model dtype
        trust_remote_code: Whether to trust remote code
        auto_inject_cache: Auto-inject KIVI cache in generate()
        **model_kwargs: Additional model loading arguments
        
    Returns:
        model: Patched Qwen model
        tokenizer: Tokenizer
        
    Example:
        model, tokenizer = load_qwen_with_kivi(
            "Qwen/Qwen2-7B-Instruct",
            k_bits=2,
            v_bits=2,
        )
        
        outputs = model.generate(
            tokenizer("Hello", return_tensors="pt").input_ids.cuda(),
            max_new_tokens=100,
        )
        print(tokenizer.decode(outputs[0]))
    """
    print(f"Loading {model_name_or_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )
    
    # Patch model
    patch_qwen_model(
        model,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
        auto_inject_cache=auto_inject_cache,
    )
    
    return model, tokenizer


# ============================================================================
# Alternative: Wrap entire model forward
# ============================================================================

class QwenKIVIWrapper(nn.Module):
    """
    Wrapper that adds KIVI cache to any Qwen model.
    
    This is an alternative to patching that wraps the entire model.
    """
    
    def __init__(
        self,
        model,
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 32,
        residual_length: int = 128,
    ):
        super().__init__()
        self.model = model
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length
        
        # Preserve model attributes
        self.config = model.config
        self.device = next(model.parameters()).device
    
    def create_kivi_cache(self) -> KIVICache:
        """Create a new KIVI cache."""
        return KIVICache(
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            group_size=self.group_size,
            residual_length=self.residual_length,
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # Create KIVI cache if needed
        if use_cache and past_key_values is None:
            past_key_values = self.create_kivi_cache()
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
    
    def generate(self, *args, **kwargs):
        """Generate with KIVI cache."""
        if 'past_key_values' not in kwargs or kwargs['past_key_values'] is None:
            kwargs['past_key_values'] = self.create_kivi_cache()
        if 'use_cache' not in kwargs:
            kwargs['use_cache'] = True
        return self.model.generate(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ============================================================================
# Utility functions
# ============================================================================

def get_kivi_memory_savings(
    seq_length: int,
    k_bits: int = 2,
    v_bits: int = 2,
    residual_length: int = 128,
) -> Dict[str, float]:
    """
    Calculate memory savings from KIVI quantization.
    
    Args:
        seq_length: Total sequence length
        k_bits, v_bits: Quantization bits
        residual_length: Full precision residual length
        
    Returns:
        Dictionary with memory statistics
    """
    if seq_length <= residual_length:
        quant_len = 0
        residual_len = seq_length
    else:
        quant_len = seq_length - residual_length
        residual_len = residual_length
    
    # Full precision bits
    fp16_bits = 16
    
    # Average bits per element
    avg_k_bits = (quant_len * k_bits + residual_len * fp16_bits) / seq_length
    avg_v_bits = (quant_len * v_bits + residual_len * fp16_bits) / seq_length
    avg_bits = (avg_k_bits + avg_v_bits) / 2
    
    return {
        "seq_length": seq_length,
        "quantized_tokens": quant_len,
        "residual_tokens": residual_len,
        "k_bits_avg": f"{avg_k_bits:.2f}",
        "v_bits_avg": f"{avg_v_bits:.2f}",
        "total_bits_avg": f"{avg_bits:.2f}",
        "memory_ratio": f"{avg_bits / fp16_bits:.2%}",
        "compression_ratio": f"{fp16_bits / avg_bits:.2f}x",
    }


def print_kivi_stats(cache: KIVICache):
    """Print KIVI cache statistics."""
    info = cache.get_cache_info()
    print("\n" + "=" * 50)
    print("KIVI Cache Statistics")
    print("=" * 50)
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "KIVICache",
    "load_qwen_with_kivi",
    "patch_qwen_model",
    "QwenKIVIWrapper",
    "get_kivi_memory_savings",
    "print_kivi_stats",
]
