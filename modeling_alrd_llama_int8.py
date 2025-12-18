"""
ALRDLlama - INT8 版本 (使用 bitsandbytes)
Value 重建使用 bitsandbytes 的 INT8 矩阵乘法
使用 FlashAttention2
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

logger = logging.get_logger(__name__)


# ============================================================================
# 检测可用的 INT8 后端
# ============================================================================

# 检查 bitsandbytes
BNB_AVAILABLE = False
bnb = None
try:
    import bitsandbytes as bnb_module
    bnb = bnb_module
    from bitsandbytes.nn import Linear8bitLt, Int8Params
    BNB_AVAILABLE = True
except ImportError:
    pass

# 检查 torch._int_mm
TORCH_INT_MM_AVAILABLE = hasattr(torch, '_int_mm')


def get_int8_backends():
    """获取可用的 INT8 后端"""
    backends = {
        'bitsandbytes': BNB_AVAILABLE,
        'torch_int_mm': TORCH_INT_MM_AVAILABLE,
        'cuda': torch.cuda.is_available(),
    }
    return backends


def print_int8_support():
    """打印 INT8 支持情况"""
    backends = get_int8_backends()
    print("=" * 60)
    print("INT8 Backend Support")
    print("=" * 60)
    print(f"bitsandbytes: {backends['bitsandbytes']}")
    print(f"torch._int_mm: {backends['torch_int_mm']}")
    print(f"CUDA: {backends['cuda']}")
    if backends['bitsandbytes']:
        print(f"  -> Using bitsandbytes (recommended)")
    elif backends['torch_int_mm']:
        print(f"  -> Using torch._int_mm")
    else:
        print(f"  -> Using fallback (no speedup)")
    print("=" * 60)


# ============================================================================
# INT8 量化工具
# ============================================================================

@torch.no_grad()
def quantize_to_int8_symmetric(x, dim=-1):
    """对称 INT8 量化"""
    amax = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


# ============================================================================
# INT8 矩阵乘法实现
# ============================================================================

def create_bnb_linear(in_features: int, out_features: int, weight: torch.Tensor, 
                       bias: Optional[torch.Tensor] = None, device: str = "cuda"):
    """
    创建 bitsandbytes 的 8-bit Linear 层
    
    Args:
        in_features: 输入维度
        out_features: 输出维度
        weight: 权重张量 (out_features, in_features)
        bias: 偏置张量 (out_features,)
        device: 设备
    """
    has_bias = bias is not None
    
    # 创建 Linear8bitLt 层
    linear = bnb.nn.Linear8bitLt(
        in_features, 
        out_features, 
        bias=has_bias,
        has_fp16_weights=False,
        threshold=0.0,  # 不使用混合精度阈值
    )
    
    # 设置权重（触发量化）
    linear.weight = bnb.nn.Int8Params(
        weight.to(device).contiguous(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    
    # 设置偏置
    if has_bias and bias is not None:
        linear.bias = torch.nn.Parameter(bias.to(device).contiguous())
    
    return linear.to(device)


def int8_matmul_torch(x_int8: torch.Tensor, w_int8: torch.Tensor,
                       x_scale: torch.Tensor, w_scale: torch.Tensor,
                       bias: Optional[torch.Tensor] = None):
    """使用 torch._int_mm 的 INT8 矩阵乘法"""
    batch_size, seq_len, in_features = x_int8.shape
    
    x_2d = x_int8.view(-1, in_features).contiguous()
    w_T = w_int8.T.contiguous()
    
    out_int32 = torch._int_mm(x_2d, w_T)
    out = out_int32.view(batch_size, seq_len, -1).float()
    out = out * x_scale * w_scale.T
    
    if bias is not None:
        out = out + bias
    
    return out


def int8_matmul_fallback(x_int8: torch.Tensor, w_int8: torch.Tensor,
                          x_scale: torch.Tensor, w_scale: torch.Tensor,
                          bias: Optional[torch.Tensor] = None):
    """Fallback: float32 模拟"""
    out = torch.matmul(x_int8.float(), w_int8.T.float())
    out = out * x_scale * w_scale.T
    if bias is not None:
        out = out + bias
    return out


# ============================================================================
# ALRDLinear INT8 版本 (bitsandbytes)
# ============================================================================

class ALRDLinearINT8(nn.Module):
    """
    低秩分解层 - INT8 版本
    
    使用 bitsandbytes 进行 INT8 矩阵乘法 (如果可用)
    """
    
    def __init__(self, in_features, out_features, rank, bias=True, backend="auto"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.backend = backend
        self.has_bias = bias
        
        # 低秩投影 B
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        
        # 重建矩阵 A (用于 FP16 和作为权重来源)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        
        # bitsandbytes 8-bit 版本 (延迟初始化)
        self.ALinear_bnb = None
        
        # Latent 量化器
        self.quantizer = Quantizer(n_bits=4, group_size=0, sym=True, clip_ratio=1.0)
        
        # INT8 权重缓存 (用于 torch._int_mm)
        self.register_buffer('A_int8', None)
        self.register_buffer('A_scale', None)
        self._int8_prepared = False
        
        # 选择后端
        self._select_backend()
    
    def _select_backend(self):
        """选择最佳后端"""
        if self.backend == "auto":
            if BNB_AVAILABLE:
                self.backend = "bnb"
            elif TORCH_INT_MM_AVAILABLE:
                self.backend = "torch"
            else:
                self.backend = "fallback"
    
    def prepare_int8_weights(self):
        """预量化权重"""
        if self._int8_prepared:
            return
        
        with torch.no_grad():
            if self.backend == "bnb" and BNB_AVAILABLE:
                # 创建 bitsandbytes 8-bit Linear
                device = self.ALinear.weight.device
                self.ALinear_bnb = create_bnb_linear(
                    self.rank, 
                    self.out_features,
                    self.ALinear.weight.data,
                    self.ALinear.bias if self.has_bias else None,
                    device=str(device)
                )
            else:
                # torch._int_mm 或 fallback
                w = self.ALinear.weight.data.float()
                w_int8, w_scale = quantize_to_int8_symmetric(w, dim=-1)
                self.A_int8 = w_int8.contiguous()
                self.A_scale = w_scale.contiguous()
            
            self._int8_prepared = True
    
    def quantize_latent(self, latents):
        return self.quantizer(latents)
    
    def quantize_latent_mixed(self, latents):
        return self.quantizer(latents)
    
    def forward_int8(self, latent):
        """INT8 版本的重建"""
        
        if not self._int8_prepared:
            self.prepare_int8_weights()
        
        if self.backend == "bnb" and BNB_AVAILABLE and self.ALinear_bnb is not None:
            # 使用 bitsandbytes Linear8bitLt
            out = self.ALinear_bnb(latent)
            return out.to(latent.dtype)
        
        elif self.backend == "torch" and TORCH_INT_MM_AVAILABLE:
            # 使用 torch._int_mm
            latent_int8, latent_scale = quantize_to_int8_symmetric(latent.float(), dim=-1)
            latent_int8 = latent_int8.contiguous()
            
            out = int8_matmul_torch(
                latent_int8, self.A_int8,
                latent_scale, self.A_scale,
                self.ALinear.bias
            )
            return out.to(latent.dtype)
        
        else:
            # Fallback: 使用 FP16
            return self.ALinear(latent)
    
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
    """INT8 版本的 Flash Attention2"""
    
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


def replace_kv_proj_with_alrd_int8(model, truncation_ranks: dict, backend: str = "auto"):
    """将 k_proj 和 v_proj 替换为低秩版本"""
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
                    backend=backend
                )
            
            setattr(parent, parts[-1], new_layer)
    return model


def prepare_all_int8_weights(model):
    """预量化所有 INT8 权重"""
    for name, module in model.named_modules():
        if isinstance(module, ALRDLinearINT8):
            if module.backend in ["torch", "fallback"]:
                module.prepare_int8_weights()
    return model


def load_model_int8(model_path: str, truncation_ranks: dict, device: str = "cuda", backend: str = "auto"):
    """从硬盘加载模型并替换为 INT8 低秩版本"""
    print(f"Loading INT8 model from {model_path}...")
    print_int8_support()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    
    model = replace_attention_with_int8(model)
    model = replace_kv_proj_with_alrd_int8(model, truncation_ranks, backend)
    model = prepare_all_int8_weights(model)
    
    model.eval()
    print(f"INT8 model loaded successfully!")
    return model


if __name__ == "__main__":
    print_int8_support()
