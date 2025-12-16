"""
简化的 KIVI 量化模型

在每次 attention forward 时直接对 Key 和 Value states 应用 KIVI fake 量化。
不依赖 KV cache，可以直接用于 PPL 测试。

KIVI 量化策略:
- Key: per-channel 量化 (沿 token 维度量化)
- Value: per-token 量化 (沿 hidden 维度量化)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel, LlamaDecoderLayer, LlamaAttention, 
    LlamaFlashAttention2, LlamaSdpaAttention,
    apply_rotary_pos_emb, repeat_kv
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)


# ============================================================================
# KIVI Fake Quantization Functions
# ============================================================================

@torch.no_grad()
def kivi_quantize_per_channel(
    x: torch.Tensor, 
    n_bits: int = 2, 
    group_size: int = 128
) -> torch.Tensor:
    """
    KIVI per-channel 量化 (用于 Key)
    沿 token 维度量化，保持每个 channel 的方差结构
    
    Args:
        x: [batch, seq_len, dim] 或 [batch, heads, seq_len, head_dim]
        n_bits: 量化位数
        group_size: 分组大小
    
    Returns:
        fake 量化后的 tensor (同形状)
    """
    if n_bits >= 16:
        return x
    
    q_max = 2 ** n_bits - 1
    q_min = 0
    
    original_shape = x.shape
    
    # 转置: [batch, seq_len, dim] -> [batch, dim, seq_len]
    x = x.transpose(-1, -2)
    
    if group_size > 0 and x.shape[-1] >= group_size:
        # 按 group 量化
        *leading_dims, seq_len = x.shape
        n_groups = seq_len // group_size
        
        if n_groups > 0:
            aligned_len = n_groups * group_size
            x_aligned = x[..., :aligned_len]
            x_remainder = x[..., aligned_len:]
            
            # Reshape to groups
            x_grouped = x_aligned.view(*leading_dims, n_groups, group_size)
            
            # Per-group min-max
            x_min = x_grouped.amin(dim=-1, keepdim=True)
            x_max = x_grouped.amax(dim=-1, keepdim=True)
            scale = (x_max - x_min).clamp(min=1e-5) / q_max
            zero_point = (-x_min / scale).round().clamp(q_min, q_max)
            
            # Quantize and dequantize
            x_quant = (x_grouped / scale + zero_point).round().clamp(q_min, q_max)
            x_dequant = (x_quant - zero_point) * scale
            
            # Reshape back
            x_dequant = x_dequant.view(*leading_dims, aligned_len)
            
            # Handle remainder (keep as-is or quantize without grouping)
            if x_remainder.shape[-1] > 0:
                x_dequant = torch.cat([x_dequant, x_remainder], dim=-1)
        else:
            x_dequant = x
    else:
        # 不分组，整体量化
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-5) / q_max
        zero_point = (-x_min / scale).round().clamp(q_min, q_max)
        
        x_quant = (x / scale + zero_point).round().clamp(q_min, q_max)
        x_dequant = (x_quant - zero_point) * scale
    
    # 转置回来
    x_dequant = x_dequant.transpose(-1, -2)
    
    return x_dequant


@torch.no_grad()
def kivi_quantize_per_token(
    x: torch.Tensor, 
    n_bits: int = 2, 
    group_size: int = 128
) -> torch.Tensor:
    """
    KIVI per-token 量化 (用于 Value)
    沿 hidden 维度量化
    
    Args:
        x: [batch, seq_len, dim] 或 [batch, heads, seq_len, head_dim]
        n_bits: 量化位数
        group_size: 分组大小
    
    Returns:
        fake 量化后的 tensor (同形状)
    """
    if n_bits >= 16:
        return x
    
    q_max = 2 ** n_bits - 1
    q_min = 0
    
    if group_size > 0 and x.shape[-1] >= group_size:
        # 按 group 量化 (沿最后一维)
        *leading_dims, dim = x.shape
        n_groups = dim // group_size
        
        if n_groups > 0:
            aligned_dim = n_groups * group_size
            x_aligned = x[..., :aligned_dim]
            x_remainder = x[..., aligned_dim:]
            
            # Reshape to groups
            x_grouped = x_aligned.view(*leading_dims, n_groups, group_size)
            
            # Per-group min-max
            x_min = x_grouped.amin(dim=-1, keepdim=True)
            x_max = x_grouped.amax(dim=-1, keepdim=True)
            scale = (x_max - x_min).clamp(min=1e-5) / q_max
            zero_point = (-x_min / scale).round().clamp(q_min, q_max)
            
            # Quantize and dequantize
            x_quant = (x_grouped / scale + zero_point).round().clamp(q_min, q_max)
            x_dequant = (x_quant - zero_point) * scale
            
            # Reshape back
            x_dequant = x_dequant.view(*leading_dims, aligned_dim)
            
            # Handle remainder
            if x_remainder.shape[-1] > 0:
                x_dequant = torch.cat([x_dequant, x_remainder], dim=-1)
        else:
            x_dequant = x
    else:
        # 不分组，整体量化
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-5) / q_max
        zero_point = (-x_min / scale).round().clamp(q_min, q_max)
        
        x_quant = (x / scale + zero_point).round().clamp(q_min, q_max)
        x_dequant = (x_quant - zero_point) * scale
    
    return x_dequant


# ============================================================================
# KIVI Config
# ============================================================================

class KIVILlamaConfig(LlamaConfig):
    """带 KIVI 量化参数的 Llama 配置"""
    
    model_type = "kivi_llama"
    
    def __init__(
        self,
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 128,
        residual_length: int = 128,  # 保留全精度的最近 token 数
        **kwargs
    ):
        super().__init__(**kwargs)
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length


# ============================================================================
# KIVI Attention (直接在 forward 中量化 K/V)
# ============================================================================

class KIVILlamaAttention(nn.Module):
    """
    KIVI 量化的 Llama Attention
    
    在每次 forward 时对 Key 和 Value states 应用 KIVI fake 量化
    """
    
    def __init__(self, config: KIVILlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # KIVI 参数
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.group_size = config.group_size
        self.residual_length = config.residual_length
    
    def _apply_kivi_quantization(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 Key 和 Value states 应用 KIVI 量化
        
        Args:
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            value_states: [batch, num_kv_heads, seq_len, head_dim]
        
        Returns:
            量化后的 key_states, value_states
        """
        seq_len = key_states.shape[2]
        
        # 保留最近的 residual_length 个 token 不量化
        if seq_len <= self.residual_length:
            # 所有 token 都保持全精度
            return key_states, value_states
        
        n_quant = seq_len - self.residual_length
        # 对齐到 group_size
        if self.group_size > 0:
            n_quant = (n_quant // self.group_size) * self.group_size
        
        if n_quant <= 0:
            return key_states, value_states
        
        # 分离要量化的部分和残差部分
        key_to_quant = key_states[:, :, :n_quant, :]
        key_residual = key_states[:, :, n_quant:, :]
        
        value_to_quant = value_states[:, :, :n_quant, :]
        value_residual = value_states[:, :, n_quant:, :]
        
        # 应用 KIVI 量化
        # Key: per-channel (沿 token 维度)
        key_quantized = kivi_quantize_per_channel(
            key_to_quant, 
            n_bits=self.k_bits, 
            group_size=self.group_size
        )
        
        # Value: per-token (沿 head_dim 维度)
        value_quantized = kivi_quantize_per_token(
            value_to_quant, 
            n_bits=self.v_bits, 
            group_size=self.group_size
        )
        
        # 合并
        key_states = torch.cat([key_quantized, key_residual], dim=2)
        value_states = torch.cat([value_quantized, value_residual], dim=2)
        
        return key_states, value_states
    
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
        
        # 计算 Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # 更新 cache (如果使用)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        
        # ================================================================
        # 应用 KIVI 量化 (核心修改)
        # ================================================================
        key_states, value_states = self._apply_kivi_quantization(key_states, value_states)
        
        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class KIVILlamaSdpaAttention(KIVILlamaAttention):
    """使用 SDPA 的 KIVI Attention"""
    
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
                hidden_states, attention_mask, position_ids, past_key_value,
                output_attentions, use_cache, cache_position, position_embeddings, **kwargs
            )
        
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        
        # ================================================================
        # 应用 KIVI 量化
        # ================================================================
        key_states, value_states = self._apply_kivi_quantization(key_states, value_states)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]
        
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


# ============================================================================
# KIVI Decoder Layer
# ============================================================================

class KIVILlamaDecoderLayer(nn.Module):
    """带 KIVI 量化的 Decoder Layer"""
    
    def __init__(self, config: KIVILlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # 使用 KIVI attention
        self.self_attn = KIVILlamaSdpaAttention(config=config, layer_idx=layer_idx)
        
        # 复用原始的 MLP 和 LayerNorm
        from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention with KIVI quantization
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


# ============================================================================
# KIVI Model
# ============================================================================

class KIVILlamaModel(LlamaModel):
    """带 KIVI 量化的 Llama Model"""
    
    def __init__(self, config: KIVILlamaConfig):
        super().__init__(config)
        
        # 替换所有 decoder layers
        self.layers = nn.ModuleList([
            KIVILlamaDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])


class KIVILlamaForCausalLM(LlamaForCausalLM):
    """
    带 KIVI 量化的 Llama Causal LM
    
    使用方法:
        config = KIVILlamaConfig.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            k_bits=2,
            v_bits=2,
            group_size=128,
            residual_length=128,
        )
        model = KIVILlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            config=config,
            torch_dtype=torch.float16,
        )
    """
    
    config_class = KIVILlamaConfig
    
    def __init__(self, config: KIVILlamaConfig):
        super().__init__(config)
        self.model = KIVILlamaModel(config)
        
        print(f"[KIVI] Initialized with k_bits={config.k_bits}, v_bits={config.v_bits}, "
              f"group_size={config.group_size}, residual_length={config.residual_length}")


# ============================================================================
# 工具函数：从现有模型创建 KIVI 模型
# ============================================================================

def convert_to_kivi_model(
    model: LlamaForCausalLM,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 128,
    residual_length: int = 128,
) -> KIVILlamaForCausalLM:
    """
    将现有的 LlamaForCausalLM 转换为 KIVI 量化版本
    
    Args:
        model: 原始 LlamaForCausalLM 模型
        k_bits: Key 量化位数
        v_bits: Value 量化位数
        group_size: 量化分组大小
        residual_length: 保持全精度的最近 token 数
    
    Returns:
        KIVILlamaForCausalLM 模型（共享权重）
    """
    # 创建 KIVI config
    config = KIVILlamaConfig(
        **model.config.to_dict(),
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
    )
    
    # 创建 KIVI 模型
    kivi_model = KIVILlamaForCausalLM(config)
    
    # 复制权重
    kivi_model.load_state_dict(model.state_dict(), strict=False)
    
    # 复制设备和 dtype
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    kivi_model = kivi_model.to(device=device, dtype=dtype)
    
    return kivi_model


# ============================================================================
# PPL 测试函数
# ============================================================================

@torch.no_grad()
def eval_ppl_simple(
    model,
    tokenizer,
    dataset: str = "wikitext2",
    seqlen: int = 2048,
    limit: int = None,
    device: str = "cuda",
):
    """
    简单的 PPL 测试
    
    由于 KIVI 量化已经内置在 attention 中，不需要特殊的 cache 处理
    """
    import os
    from tqdm import tqdm
    from datasets import load_dataset
    
    model = model.to(device)
    model.eval()
    
    # 加载数据
    if dataset == "wikitext2":
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen
    
    if limit is not None:
        nsamples = min(nsamples, limit)
    
    print(f"Evaluating {nsamples} samples, seqlen={seqlen}")
    
    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    
    for i in tqdm(range(nsamples), desc="Evaluating PPL"):
        batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(device)
        
        # 直接 forward，KIVI 量化会在 attention 中自动应用
        outputs = model(batch)
        logits = outputs.logits
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        nlls.append(loss.float() * (seqlen - 1))
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
    
    print(f"\n{dataset} PPL: {ppl.item():.4f}")
    return ppl.item()


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("""
使用示例:

# 方法 1: 从头加载 KIVI 模型
from modeling_kivi_simple import KIVILlamaConfig, KIVILlamaForCausalLM
from transformers import AutoTokenizer

config = KIVILlamaConfig.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    k_bits=2,
    v_bits=2,
    group_size=128,
    residual_length=128,
)

model = KIVILlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 测试 PPL
from modeling_kivi_simple import eval_ppl_simple
ppl = eval_ppl_simple(model, tokenizer, dataset="wikitext2", limit=100)


# 方法 2: 从现有模型转换
from transformers import LlamaForCausalLM
from modeling_kivi_simple import convert_to_kivi_model

# 先加载原始模型
base_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 转换为 KIVI 模型
kivi_model = convert_to_kivi_model(
    base_model,
    k_bits=2,
    v_bits=2,
    group_size=128,
    residual_length=128,
)

# 测试 PPL
ppl = eval_ppl_simple(kivi_model, tokenizer, limit=100)
""")
