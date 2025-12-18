"""
ALRDLlama - INT8 版本
Value 重建使用真正的 INT8 矩阵乘法
使用 FlashAttention2

支持多种 INT8 后端:
1. torch._int_mm (PyTorch 2.0+, CUDA)
2. Fallback: float32 模拟 (无速度提升，仅用于精度测试)
"""
from transformers import LlamaForCausalLM, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaFlashAttention2, 
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
from int8_ops import INT8Quantizer, INT8Linear, INT8LinearFunction, TRITON_AVAILABLE

logger = logging.get_logger(__name__)


# ============================================================================
# INT8 后端检测
# ============================================================================

def check_int8_support():
    """检测可用的 INT8 后端"""
    backends = {
        'triton': TRITON_AVAILABLE,
        'torch_int_mm': hasattr(torch, '_int_mm'),
        'cuda_available': torch.cuda.is_available(),
    }
    
    if backends['torch_int_mm'] and torch.cuda.is_available():
        try:
            a = torch.randint(-128, 127, (32, 64), dtype=torch.int8, device='cuda')
            b = torch.randint(-128, 127, (64, 32), dtype=torch.int8, device='cuda')
            _ = torch._int_mm(a, b)
            backends['torch_int_mm_tested'] = True
        except Exception as e:
            backends['torch_int_mm_tested'] = False
            backends['torch_int_mm_error'] = str(e)
    
    return backends


_INT8_BACKENDS = None

def get_int8_backends():
    global _INT8_BACKENDS
    if _INT8_BACKENDS is None:
        _INT8_BACKENDS = check_int8_support()
    return _INT8_BACKENDS


# ============================================================================
# ALRDLinear INT8 版本
# ============================================================================

class ALRDLinearINT8(nn.Module):
    """
    低秩分解层 - INT8 版本 (Value 重建使用真正的 INT8 量化)
    
    使用 INT8Quantizer:
    - Value latent: per-token INT8 量化 (在线)
    - 重建矩阵 A: per-channel INT8 量化 (预量化)
    
    量化流程:
    1. latent = BLinear(x)
    2. latent_int8, latent_scale = quantize_per_token(latent)
    3. out_int32 = latent_int8 @ A_int8.T
    4. out = dequantize(out_int32, latent_scale, A_scale)
    """
    
    def __init__(self, in_features, out_features, rank, bias=True, backend="auto"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.backend = backend  # "triton", "torch", "auto"
        
        # 低秩投影 B
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        
        # 重建矩阵 A (会被量化)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        
        # Latent 量化器 (fake quant for BLinear output)
        self.quantizer = Quantizer(n_bits=4, group_size=0, sym=True, clip_ratio=1.0)
        
        # INT8 预量化的权重
        self.register_buffer('A_int8', None)
        self.register_buffer('A_scale', None)
        self._int8_prepared = False
    
    def prepare_int8_weights(self):
        """预量化 ALinear 权重为 INT8 (per-channel)"""
        if self._int8_prepared:
            return
        
        with torch.no_grad():
            w = self.ALinear.weight.data  # (out_features, rank)
            # 使用 INT8Quantizer 进行 per-channel 量化
            w_int8, w_scale = INT8Quantizer.quantize_per_channel(w.float())
            
            self.A_int8 = w_int8.contiguous()
            self.A_scale = w_scale.contiguous()
            self._int8_prepared = True
    
    def quantize_latent(self, latents):
        """对 latent 进行 fake quant (为了模拟低精度存储)"""
        return self.quantizer(latents)
    
    def quantize_latent_mixed(self, latents):
        return self.quantizer(latents)
    
    def forward_int8(self, latent):
        """
        INT8 版本的重建
        
        Args:
            latent: (batch, seq, rank) - 量化后的 latent (fake quant float)
        
        Returns:
            out: (batch, seq, out_features)
        """
        if not self._int8_prepared:
            self.prepare_int8_weights()
        
        # 使用 INT8LinearFunction 进行 INT8 矩阵乘法
        # 内部会: 量化 latent (per-token) -> INT8 matmul -> 反量化
        out = INT8LinearFunction.apply(
            latent,              # 输入
            self.A_int8,         # 预量化的权重 (int8)
            self.A_scale,        # 权重 scale
            self.ALinear.bias,   # 偏置
            self.backend         # 后端选择
        )
        
        return out.to(latent.dtype)
    
    def forward(self, x):
        """默认使用 FP16"""
        y = self.BLinear(x)
        y = self.quantizer(y)
        return self.ALinear(y)


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
# 自定义 FlashAttention2
# ============================================================================

class CustomLlamaFlashAttention2INT8(LlamaFlashAttention2):
    """INT8 版本的 Flash Attention2 - Value 重建使用 INT8"""
    
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
        
        # Key & Value: 低秩投影 → 量化
        key_states = self.k_proj.BLinear(hidden_states)
        key_states = self.k_proj.quantize_latent(key_states)
        value_states = self.v_proj.BLinear(hidden_states)
        value_states = self.v_proj.quantize_latent(value_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 更新 KV Cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        
        # Key: FP16 重建, Value: INT8 重建
        key_states = self.k_proj.ALinear(key_states)
        value_states = self.v_proj.forward_int8(value_states)
        
        _, k_len, _ = key_states.size()
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        if cached_position_embeddings is not None:
            cos, sin = cached_position_embeddings
        else:
            cos, sin = position_embeddings
            
        if q_len > 1:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos[:, -1:, :], sin[:, -1:, :])
            key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        
        # Flash Attention 格式
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
# 模型加载函数
# ============================================================================

def replace_attention_with_int8(model):
    """将模型中的 attention 替换为 INT8 版本"""
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            old_attn = module.self_attn
            new_attn = CustomLlamaFlashAttention2INT8(old_attn.config, old_attn.layer_idx)
            new_attn.q_proj = old_attn.q_proj
            new_attn.k_proj = old_attn.k_proj
            new_attn.v_proj = old_attn.v_proj
            new_attn.o_proj = old_attn.o_proj
            module.self_attn = new_attn
    return model


def replace_kv_proj_with_alrd_int8(model, truncation_ranks: dict, use_native_int8: bool = True):
    """将 k_proj 和 v_proj 替换为低秩版本 (v_proj 使用 INT8)"""
    for name, module in model.named_modules():
        if name in truncation_ranks:
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            old_linear = getattr(parent, parts[-1])
            
            if 'k_proj' in name:
                new_layer = ALRDLinearKeyFP16(
                    old_linear.in_features,
                    old_linear.out_features,
                    truncation_ranks[name],
                    bias=old_linear.bias is not None
                )
            else:  # v_proj
                new_layer = ALRDLinearINT8(
                    old_linear.in_features,
                    old_linear.out_features,
                    truncation_ranks[name],
                    bias=old_linear.bias is not None,
                    use_native_int8=use_native_int8
                )
            
            setattr(parent, parts[-1], new_layer)
    return model


def prepare_all_int8_weights(model):
    """预量化所有 INT8 权重"""
    backends = get_int8_backends()
    use_native = backends.get('torch_int_mm_tested', False)
    print(f"INT8 backend: {'torch._int_mm' if use_native else 'float32 fallback'}")
    
    for name, module in model.named_modules():
        if isinstance(module, ALRDLinearINT8):
            module.prepare_int8_weights()
    return model


def load_model_int8(model_path: str, truncation_ranks: dict, device: str = "cuda", use_native_int8: bool = True):
    """
    从硬盘加载模型并替换为 INT8 低秩版本
    
    Args:
        model_path: 模型路径
        truncation_ranks: 低秩配置 {layer_name: rank}
        device: 设备
        use_native_int8: 是否使用原生 INT8 kernel
    """
    print(f"Loading INT8 model from {model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    
    model = replace_attention_with_int8(model)
    model = replace_kv_proj_with_alrd_int8(model, truncation_ranks, use_native_int8)
    model = prepare_all_int8_weights(model)
    
    model.eval()
    print(f"INT8 model loaded successfully!")
    return model


# ============================================================================
# 工具函数
# ============================================================================

def print_int8_support():
    """打印 INT8 支持情况"""
    backends = get_int8_backends()
    print("=" * 60)
    print("INT8 Backend Support")
    print("=" * 60)
    print(f"CUDA available: {backends['cuda_available']}")
    print(f"torch._int_mm available: {backends['torch_int_mm']}")
    if 'torch_int_mm_tested' in backends:
        print(f"torch._int_mm tested: {backends['torch_int_mm_tested']}")
    print("=" * 60)


if __name__ == "__main__":
    print_int8_support()
