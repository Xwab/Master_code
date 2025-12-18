"""
ALRDLlama - INT8 版本
Value 重建使用 INT8 矩阵乘法 (模拟)
- Value latent: per-token INT8 量化
- ALinear.weight: per-channel INT8 量化 (预量化)
- 使用 INT32 累加后反量化
"""
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaFlashAttention2, LlamaSdpaAttention, 
    LlamaModel, repeat_kv, apply_rotary_pos_emb
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List

from modules.quant_utils import Quantizer

logger = logging.get_logger(__name__)


# ============================================================================
# INT8 量化工具
# ============================================================================

@torch.no_grad()
def quantize_to_int8_symmetric(x, dim=-1):
    """
    对称 INT8 量化
    Args:
        x: 输入 tensor
        dim: 量化维度
    Returns:
        x_int8: INT8 量化结果
        scale: 缩放因子
    """
    amax = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


class INT8MatmulFunction(torch.autograd.Function):
    """INT8 矩阵乘法的自定义函数"""
    
    @staticmethod
    def forward(ctx, x, w_int8, w_scale):
        """
        x: (batch, seq, in_features) - FP16 输入
        w_int8: (out_features, in_features) - INT8 预量化权重
        w_scale: (out_features, 1) - 权重缩放因子
        """
        # 量化输入 (per-token)
        x_int8, x_scale = quantize_to_int8_symmetric(x.float(), dim=-1)
        
        # INT8 @ INT8 -> INT32
        out_int32 = torch.matmul(x_int8.int(), w_int8.T.int())
        
        # 反量化
        out = out_int32.float() * x_scale * w_scale.T
        
        return out.to(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None


def int8_matmul(x, w_int8, w_scale):
    """INT8 矩阵乘法封装"""
    return INT8MatmulFunction.apply(x, w_int8, w_scale)


# ============================================================================
# ALRDLinear INT8 版本
# ============================================================================

class ALRDLinearINT8(nn.Module):
    """低秩分解层 - INT8 版本 (Value 重建使用 INT8)"""
    
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # 低秩投影
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        
        # 重建矩阵 - 将被替换为 INT8
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        
        # Latent 量化器
        self.quantizer = Quantizer(n_bits=4, group_size=0, sym=True, clip_ratio=1.0)
        
        # INT8 权重 (初始化后需要调用 prepare_int8_weights)
        self.register_buffer('A_int8', None)
        self.register_buffer('A_scale', None)
        self._int8_prepared = False
    
    def prepare_int8_weights(self):
        """预量化 ALinear 权重为 INT8"""
        if self._int8_prepared:
            return
        
        with torch.no_grad():
            w = self.ALinear.weight.data.float()  # (out_features, rank)
            w_int8, w_scale = quantize_to_int8_symmetric(w, dim=-1)
            
            self.A_int8 = w_int8
            self.A_scale = w_scale
            self._int8_prepared = True
    
    def quantize_latent(self, latents):
        return self.quantizer(latents)
    
    def quantize_latent_mixed(self, latents):
        return self.quantizer(latents)
    
    def forward_fp16(self, x):
        """FP16 版本的 forward (用于 Key)"""
        y = self.BLinear(x)
        y = self.quantizer(y)
        return self.ALinear(y)
    
    def forward_int8(self, latent):
        """INT8 版本的重建 (用于 Value)"""
        if not self._int8_prepared:
            self.prepare_int8_weights()
        
        # INT8 矩阵乘法
        out = int8_matmul(latent, self.A_int8, self.A_scale)
        
        # 加 bias
        if self.ALinear.bias is not None:
            out = out + self.ALinear.bias
        
        return out
    
    def forward(self, x):
        """默认使用 FP16"""
        return self.forward_fp16(x)


class ALRDLinearKeyFP16(nn.Module):
    """Key 投影层 - 使用 FP16"""
    
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.rank = rank
        self.quantizer = Quantizer(n_bits=4, group_size=0, sym=True, clip_ratio=1.0)
    
    def quantize_latent(self, latents):
        return self.quantizer(latents)
    
    def quantize_latent_mixed(self, latents):
        return self.quantizer(latents)
    
    def forward(self, x):
        y = self.BLinear(x)
        y = self.quantizer(y)
        return self.ALinear(y)


# ============================================================================
# 自定义 Attention 层
# ============================================================================

class CustomLlamaSdpaAttentionINT8(LlamaSdpaAttention):
    """INT8 版本的 SDPA Attention - Value 重建使用 INT8"""
    
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)

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
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        
        cached_position_embeddings = kwargs.get("cached_position_embeddings", None)
        bsz, q_len, _ = hidden_states.size()
        
        # Query: 正常计算
        query_states = self.q_proj(hidden_states)
        
        # Key: 低秩投影 → 量化 (FP16 重建)
        key_states = self.k_proj.BLinear(hidden_states)
        key_states = self.k_proj.quantize_latent(key_states)
        
        # Value: 低秩投影 → 量化 (INT8 重建)
        value_states = self.v_proj.BLinear(hidden_states)
        value_states = self.v_proj.quantize_latent(value_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 更新 KV Cache (存储低秩表示)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        
        # Key: FP16 重建
        key_states = self.k_proj.ALinear(key_states)
        
        # Value: INT8 重建 !!!
        value_states = self.v_proj.forward_int8(value_states)
        
        _, k_len, _ = key_states.size()
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE
        if cached_position_embeddings is not None:
            cos, sin = cached_position_embeddings
        else:
            cos, sin = position_embeddings
            
        if q_len > 1:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos[:, -1:, :], sin[:, -1:, :])
            key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        
        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention mask
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]
        
        # SDPA
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


class CustomLlamaFlashAttention2INT8(LlamaFlashAttention2):
    """INT8 版本的 Flash Attention - Value 重建使用 INT8"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if isinstance(past_key_value, StaticCache):
            raise ValueError("StaticCache not compatible with flash_attention_2")
        
        output_attentions = False
        cached_position_embeddings = kwargs.get("cached_position_embeddings", None)
        
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj.BLinear(hidden_states)
        key_states = self.k_proj.quantize_latent(key_states)
        value_states = self.v_proj.BLinear(hidden_states)
        value_states = self.v_proj.quantize_latent(value_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        
        # Key: FP16, Value: INT8
        key_states = self.k_proj.ALinear(key_states)
        value_states = self.v_proj.forward_int8(value_states)
        
        _, k_len, _ = key_states.size()
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if cached_position_embeddings is not None:
            cos, sin = cached_position_embeddings
        else:
            cos, sin = position_embeddings
            
        if q_len > 1:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos[:, -1:, :], sin[:, -1:, :])
            key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        dropout_rate = self.attention_dropout if self.training else 0.0
        
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        
        attn_output = _flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len,
            position_ids=position_ids, dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
        
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


# ============================================================================
# 自定义模型
# ============================================================================

class CustomLlamaDecoderLayerINT8(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = CustomLlamaSdpaAttentionINT8(config, layer_idx)


class CustomLlamaModelINT8(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayerINT8(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))

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
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        cached_position = torch.arange(
            0, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cached_position_embeddings = self.rotary_emb(hidden_states, cached_position)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cached_position_embeddings=cached_position_embeddings,
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


class ALRDLlamaForCausalLMINT8(LlamaForCausalLM):
    """INT8 版本的 ALRD LLaMA"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModelINT8(config)
    
    def replace_with_alrd(self, truncation_ranks: dict):
        """将 k_proj 和 v_proj 替换为低秩版本"""
        full_name_dict = {module: name for name, module in self.named_modules()}
        linear_info = {}
        modules = [self]
        
        while len(modules) > 0:
            submodule = modules.pop()
            for name, raw_linear in submodule.named_children():
                if isinstance(raw_linear, nn.Linear):
                    full_name = full_name_dict[raw_linear]
                    linear_info[raw_linear] = {
                        "father": submodule,
                        "name": name,
                        "full_name": full_name,
                    }
                else:
                    modules.append(raw_linear)
        
        for name, module in self.named_modules():
            if name in truncation_ranks:
                info = linear_info[module]
                if 'k_proj' in name:
                    # Key: 使用 FP16 版本
                    new_layer = ALRDLinearKeyFP16(
                        module.in_features,
                        module.out_features,
                        truncation_ranks[name],
                        bias=module.bias is not None
                    )
                else:
                    # Value: 使用 INT8 版本
                    new_layer = ALRDLinearINT8(
                        module.in_features,
                        module.out_features,
                        truncation_ranks[name],
                        bias=module.bias is not None
                    )
                setattr(info["father"], info["name"], new_layer)
        
        return self
    
    def prepare_int8_weights(self):
        """预量化所有 INT8 权重"""
        for name, module in self.named_modules():
            if isinstance(module, ALRDLinearINT8):
                module.prepare_int8_weights()
        return self
