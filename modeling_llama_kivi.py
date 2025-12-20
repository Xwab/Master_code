"""
Llama Model with KIVI Quantized KV Cache (No Low-Rank Decomposition)

This module modifies Llama to use KIVI-style KV cache quantization:
- Key: per-channel quantization (along token dimension)
- Value: per-token quantization (along head_dim dimension)
- Recent tokens kept in full precision (residual)

No low-rank decomposition - uses original full model parameters.

Usage:
    from modeling_llama_kivi import LlamaForCausalLM_KIVI, LlamaKIVIConfig
    
    config = LlamaKIVIConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    config.k_bits = 2
    config.v_bits = 2
    model = LlamaForCausalLM_KIVI.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", config=config)
    
    # Generate with KIVI cache
    cache = model.create_kivi_cache()
    outputs = model.generate(input_ids, past_key_values=cache, use_cache=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
import math

# Import KIVI components
from modules.kivi_cache_general import KIVICache, KIVIQuantizer

logger = logging.get_logger(__name__)


class LlamaKIVIConfig(LlamaConfig):
    """
    Configuration for Llama with KIVI quantized KV cache.
    
    Note: We keep model_type = "llama" to avoid warnings when loading pretrained models.
    KIVI is identified by the presence of k_bits/v_bits attributes.
    """
    # Keep original model_type to avoid "model type mismatch" warnings
    model_type = "llama"
    
    def __init__(
        self,
        # KIVI parameters
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 32,
        residual_length: int = 128,
        use_kivi: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.use_kivi = use_kivi


class LlamaKIVISdpaAttention(LlamaSdpaAttention):
    """
    Llama SDPA Attention with KIVI quantized KV cache.
    
    Uses original full k_proj and v_proj - no low-rank decomposition.
    KV states are always quantized using KIVI-style quantization.
    """
    
    def __init__(self, config: LlamaKIVIConfig, layer_idx: int = None):
        super().__init__(config, layer_idx)
        
        # KIVI parameters
        self.k_bits = getattr(config, 'k_bits', 2)
        self.v_bits = getattr(config, 'v_bits', 2)
        self.group_size = getattr(config, 'group_size', 32)
        self.residual_length = getattr(config, 'residual_length', 128)
        self.use_kivi = getattr(config, 'use_kivi', True)
        
        # Create quantizers for when there's no cache
        self.key_quantizer = KIVIQuantizer(
            n_bits=self.k_bits, 
            group_size=self.group_size, 
            per_channel=True  # Key: per-channel
        )
        self.value_quantizer = KIVIQuantizer(
            n_bits=self.v_bits, 
            group_size=self.group_size, 
            per_channel=False  # Value: per-token
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if output_attentions:
            logger.warning_once(
                "LlamaKIVISdpaAttention does not support output_attentions=True. "
                "Falling back to eager attention."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        
        bsz, q_len, _ = hidden_states.size()
        
        # Standard Q, K, V projections (full rank, no decomposition)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # KIVI cache handling - following official KIVI implementation
        # Key insight: prefill uses full precision, quantization only affects storage
        
        # Save original K/V for prefill attention (before any quantization)
        key_states_for_attn = key_states
        value_states_for_attn = value_states
        
        if past_key_value is not None:
            # Check if this is prefill (cache is empty) or decode (cache has data)
            is_prefill = past_key_value.get_seq_length(self.layer_idx) == 0
            
            if is_prefill:
                # PREFILL: Use original FP16 K/V for attention, then store quantized
                # This matches official KIVI behavior
                key_states_for_attn = key_states
                value_states_for_attn = value_states
                
                # Update cache (quantize for storage, but don't use quantized for current attention)
                _, _ = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=None
                )
            else:
                # DECODE: Get cached K/V (includes quantized history + new token)
                key_states_for_attn, value_states_for_attn = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=None
                )
        elif self.use_kivi:
            # No cache but still apply KIVI quantization (for evaluation without generate)
            key_states_for_attn = self.key_quantizer(key_states)
            value_states_for_attn = self.value_quantizer(value_states)
        
        # Repeat K/V for GQA
        key_states_for_attn = repeat_kv(key_states_for_attn, self.num_key_value_groups)
        value_states_for_attn = repeat_kv(value_states_for_attn, self.num_key_value_groups)
        
        # Prepare attention mask
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, :key_states_for_attn.shape[-2]]
        
        # SDPA
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states_for_attn = key_states_for_attn.contiguous()
            value_states_for_attn = value_states_for_attn.contiguous()
        
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states_for_attn,
            value_states_for_attn,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


class LlamaKIVIFlashAttention2(LlamaFlashAttention2):
    """
    Llama Flash Attention with KIVI quantized KV cache.
    """
    
    def __init__(self, config: LlamaKIVIConfig, layer_idx: int = None):
        super().__init__(config, layer_idx)
        
        # KIVI parameters
        self.k_bits = getattr(config, 'k_bits', 2)
        self.v_bits = getattr(config, 'v_bits', 2)
        self.group_size = getattr(config, 'group_size', 32)
        self.residual_length = getattr(config, 'residual_length', 128)
        self.use_kivi = getattr(config, 'use_kivi', True)
        
        # Create quantizers for when there's no cache
        self.key_quantizer = KIVIQuantizer(
            n_bits=self.k_bits, 
            group_size=self.group_size, 
            per_channel=True  # Key: per-channel
        )
        self.value_quantizer = KIVIQuantizer(
            n_bits=self.v_bits, 
            group_size=self.group_size, 
            per_channel=False  # Value: per-token
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "FlashAttention2 is not compatible with StaticCache."
            )
        
        output_attentions = False
        
        bsz, q_len, _ = hidden_states.size()
        
        # Standard Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # KIVI cache handling - following official KIVI implementation
        # Key insight: prefill uses full precision, quantization only affects storage
        
        # Save original K/V for prefill attention (before any quantization)
        key_states_for_attn = key_states
        value_states_for_attn = value_states
        
        if past_key_value is not None:
            # Check if this is prefill (cache is empty) or decode (cache has data)
            is_prefill = past_key_value.get_seq_length(self.layer_idx) == 0
            
            if is_prefill:
                # PREFILL: Use original FP16 K/V for attention, then store quantized
                # This matches official KIVI behavior
                key_states_for_attn = key_states
                value_states_for_attn = value_states
                
                # Update cache (quantize for storage, but don't use quantized for current attention)
                _, _ = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=None
                )
            else:
                # DECODE: Get cached K/V (includes quantized history + new token)
                key_states_for_attn, value_states_for_attn = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=None
                )
        elif self.use_kivi:
            # No cache but still apply KIVI quantization (for evaluation without generate)
            key_states_for_attn = self.key_quantizer(key_states)
            value_states_for_attn = self.value_quantizer(value_states)
        
        # Flash attention requires [bsz, seq_len, num_heads, head_dim]
        # Note: Flash Attention 2 natively supports GQA, so no need for repeat_kv
        query_states = query_states.transpose(1, 2)
        key_states_for_attn = key_states_for_attn.transpose(1, 2)
        value_states_for_attn = value_states_for_attn.transpose(1, 2)
        
        dropout_rate = self.attention_dropout if self.training else 0.0
        
        # Handle dtype
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )
            
            query_states = query_states.to(target_dtype)
            key_states_for_attn = key_states_for_attn.to(target_dtype)
            value_states_for_attn = value_states_for_attn.to(target_dtype)
        
        # Import flash attention
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
        
        attn_output = _flash_attention_forward(
            query_states,
            key_states_for_attn,
            value_states_for_attn,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
        
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


class LlamaKIVIAttention(LlamaAttention):
    """
    Llama Eager Attention with KIVI quantized KV cache.
    Fallback for when SDPA/Flash is not available.
    """
    
    def __init__(self, config: LlamaKIVIConfig, layer_idx: int = None):
        super().__init__(config, layer_idx)
        
        # KIVI parameters
        self.k_bits = getattr(config, 'k_bits', 2)
        self.v_bits = getattr(config, 'v_bits', 2)
        self.group_size = getattr(config, 'group_size', 32)
        self.residual_length = getattr(config, 'residual_length', 128)
        self.use_kivi = getattr(config, 'use_kivi', True)
        
        # Create quantizers for when there's no cache
        self.key_quantizer = KIVIQuantizer(
            n_bits=self.k_bits, 
            group_size=self.group_size, 
            per_channel=True  # Key: per-channel
        )
        self.value_quantizer = KIVIQuantizer(
            n_bits=self.v_bits, 
            group_size=self.group_size, 
            per_channel=False  # Value: per-token
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Standard Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Update KV cache or apply KIVI quantization directly
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs=None
            )
        elif self.use_kivi:
            # No cache but still apply KIVI quantization
            key_states = self.key_quantizer(key_states)
            value_states = self.value_quantizer(value_states)
        
        # Repeat K/V for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class LlamaKIVIDecoderLayer(LlamaDecoderLayer):
    """
    Llama Decoder Layer with KIVI attention.
    """
    
    def __init__(self, config: LlamaKIVIConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # Replace attention with KIVI version
        attn_implementation = getattr(config, '_attn_implementation', 'sdpa')
        
        if attn_implementation == "flash_attention_2":
            self.self_attn = LlamaKIVIFlashAttention2(config, layer_idx)
        elif attn_implementation == "sdpa":
            self.self_attn = LlamaKIVISdpaAttention(config, layer_idx)
        elif attn_implementation == "eager":
            self.self_attn = LlamaKIVIAttention(config, layer_idx)
        else:
            # Default to SDPA
            self.self_attn = LlamaKIVISdpaAttention(config, layer_idx)


class LlamaKIVIModel(LlamaModel):
    """
    Llama Model with KIVI decoder layers.
    """
    
    def __init__(self, config: LlamaKIVIConfig):
        super().__init__(config)
        
        # Replace decoder layers with KIVI versions
        self.layers = nn.ModuleList(
            [LlamaKIVIDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Handle legacy cache format
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated."
                )
        
        # Setup cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        # Create causal mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        
        hidden_states = inputs_embeds
        
        # Create position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM_KIVI(LlamaForCausalLM):
    """
    Llama for Causal LM with KIVI quantized KV cache.
    
    This model uses KIVI-style quantization for the KV cache:
    - Key: per-channel quantization (along token dimension)
    - Value: per-token quantization (along head_dim dimension)
    - Recent tokens kept in full precision (residual)
    
    No low-rank decomposition - uses original full model parameters.
    
    Usage:
        # Load with KIVI config
        config = LlamaKIVIConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        config.k_bits = 2
        config.v_bits = 2
        model = LlamaForCausalLM_KIVI.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            config=config,
        )
        
        # Method 1: Create KIVI cache manually
        cache = model.create_kivi_cache()
        outputs = model.generate(input_ids, past_key_values=cache, use_cache=True)
        
        # Method 2: Auto-inject KIVI cache (enable with model.use_kivi_cache = True)
        model.use_kivi_cache = True
        outputs = model.generate(input_ids, use_cache=True)
    """
    
    config_class = LlamaKIVIConfig
    
    def __init__(self, config: LlamaKIVIConfig):
        super().__init__(config)
        
        # Replace model with KIVI version
        self.model = LlamaKIVIModel(config)
        
        # Store KIVI parameters
        self.k_bits = getattr(config, 'k_bits', 2)
        self.v_bits = getattr(config, 'v_bits', 2)
        self.group_size = getattr(config, 'group_size', 32)
        self.residual_length = getattr(config, 'residual_length', 128)
        self.use_kivi = getattr(config, 'use_kivi', True)
        
        # Flag to auto-inject KIVI cache (default: False)
        self.use_kivi_cache = False
    
    def create_kivi_cache(self) -> KIVICache:
        """
        Create a KIVI cache for this model.
        
        Returns:
            KIVICache configured with model's KIVI parameters
            
        Usage:
            cache = model.create_kivi_cache()
            outputs = model.generate(input_ids, past_key_values=cache, use_cache=True)
        """
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with optional auto-injection of KIVI cache.
        
        Set model.use_kivi_cache = True to auto-inject KIVI cache.
        """
        use_cache_flag = use_cache if use_cache is not None else self.config.use_cache
        
        # Auto-inject KIVI cache if enabled
        if self.use_kivi_cache and self.use_kivi and use_cache_flag:
            if past_key_values is None or not isinstance(past_key_values, KIVICache):
                past_key_values = self.create_kivi_cache()
        
        return super().forward(
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
            cache_position=cache_position,
            **kwargs,
        )
    
    def generate(
        self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        """
        Generate with optional auto-injection of KIVI cache.
        
        Set model.use_kivi_cache = True to auto-inject KIVI cache.
        """
        # Auto-inject KIVI cache if enabled
        if self.use_kivi_cache and self.use_kivi:
            if kwargs.get('past_key_values') is None or not isinstance(kwargs.get('past_key_values'), KIVICache):
                kwargs['past_key_values'] = self.create_kivi_cache()
                print(f"[KIVI] Auto-created KIVICache (k={self.k_bits}bit, v={self.v_bits}bit)")
            
            if 'use_cache' not in kwargs:
                kwargs['use_cache'] = True
        
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load pretrained model with KIVI support.
        
        KIVI parameters can be set via config:
            config = LlamaKIVIConfig.from_pretrained(model_path)
            config.k_bits = 2
            config.v_bits = 2
            model = LlamaForCausalLM_KIVI.from_pretrained(model_path, config=config)
        
        Or via kwargs:
            model = LlamaForCausalLM_KIVI.from_pretrained(
                model_path,
                k_bits=2,
                v_bits=2,
            )
        """
        config = kwargs.get('config')
        
        if config is None:
            # Load config and convert to KIVI config
            config = LlamaKIVIConfig.from_pretrained(pretrained_model_name_or_path)
        elif not isinstance(config, LlamaKIVIConfig):
            # Convert to KIVI config
            config_dict = config.to_dict()
            config = LlamaKIVIConfig(**config_dict)
        
        # Apply KIVI parameters from kwargs
        if 'k_bits' in kwargs:
            config.k_bits = kwargs.pop('k_bits')
        if 'v_bits' in kwargs:
            config.v_bits = kwargs.pop('v_bits')
        if 'group_size' in kwargs:
            config.group_size = kwargs.pop('group_size')
        if 'residual_length' in kwargs:
            config.residual_length = kwargs.pop('residual_length')
        if 'use_kivi' in kwargs:
            config.use_kivi = kwargs.pop('use_kivi')
        
        kwargs['config'] = config
        
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


# ============================================================================
# Utility Functions
# ============================================================================

def load_llama_kivi(
    model_name_or_path: str,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    **kwargs,
) -> Tuple[LlamaForCausalLM_KIVI, Any]:
    """
    Load Llama model with KIVI cache support.
    
    Args:
        model_name_or_path: Model path or name
        k_bits: Key quantization bits
        v_bits: Value quantization bits
        group_size: Quantization group size
        residual_length: Full precision residual length
        torch_dtype: Model dtype
        device_map: Device placement
        **kwargs: Additional arguments for from_pretrained
        
    Returns:
        model: LlamaForCausalLM_KIVI
        tokenizer: Tokenizer
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    model = LlamaForCausalLM_KIVI.from_pretrained(
        model_name_or_path,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        **kwargs,
    )
    
    print(f"[KIVI] Loaded {model_name_or_path} with KIVI support")
    print(f"[KIVI] k_bits={k_bits}, v_bits={v_bits}, group_size={group_size}, residual={residual_length}")
    
    return model, tokenizer


def get_kivi_memory_stats(cache: KIVICache) -> Dict[str, Any]:
    """Get KIVI cache memory statistics."""
    return cache.get_cache_info()


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
    "LlamaKIVIConfig",
    "LlamaForCausalLM_KIVI",
    "LlamaKIVIModel",
    "LlamaKIVISdpaAttention",
    "LlamaKIVIFlashAttention2",
    "LlamaKIVIAttention",
    "LlamaKIVIDecoderLayer",
    "KIVICache",
    "load_llama_kivi",
    "get_kivi_memory_stats",
    "print_kivi_stats",
]
