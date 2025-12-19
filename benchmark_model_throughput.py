"""
INT8 vs FP16 完整模型吞吐量对比 Benchmark

测试完整 LLaMA 模型的 prefill 和 decode throughput
使用随机初始化权重（不需要本地模型）
"""

import torch
import torch.nn as nn
import time
import argparse
import gc
from typing import Optional, Tuple, List
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaFlashAttention2, LlamaAttention,
    apply_rotary_pos_emb
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.cache_utils import DynamicCache

# ============================================================================
# 检测可用的 INT8 后端
# ============================================================================

BNB_AVAILABLE = False
bnb = None
try:
    import bitsandbytes as bnb_module
    bnb = bnb_module
    BNB_AVAILABLE = True
except ImportError:
    pass


def check_backends():
    """检测可用的后端"""
    print("=" * 70)
    print("Backend Detection")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"bitsandbytes: {BNB_AVAILABLE}", end="")
    if BNB_AVAILABLE:
        print(f" (v{bnb.__version__})")
    else:
        print()
    print(f"torch._int_mm: {hasattr(torch, '_int_mm')}")
    print("=" * 70)


# ============================================================================
# 量化工具
# ============================================================================

class Quantizer(nn.Module):
    """简单的 fake quantizer"""
    def __init__(self, n_bits=4):
        super().__init__()
        self.n_bits = n_bits
        self.qmax = 2 ** (n_bits - 1) - 1
        self.qmin = -(2 ** (n_bits - 1))
    
    def forward(self, x):
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.qmax
        x_q = (x / scale).round().clamp(self.qmin, self.qmax)
        return x_q * scale


def create_bnb_linear(in_features: int, out_features: int, weight: torch.Tensor, 
                       bias: Optional[torch.Tensor] = None, device: str = "cuda"):
    """创建 bitsandbytes 的 8-bit Linear 层"""
    if not BNB_AVAILABLE:
        raise RuntimeError("bitsandbytes not available")
    
    has_bias = bias is not None
    
    linear = bnb.nn.Linear8bitLt(
        in_features, 
        out_features, 
        bias=has_bias,
        has_fp16_weights=False,
        threshold=0.0,
    )
    
    linear.weight = bnb.nn.Int8Params(
        weight.to(device).contiguous(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    
    if has_bias and bias is not None:
        linear.bias = nn.Parameter(bias.to(device).contiguous())
    
    return linear.to(device)


# ============================================================================
# 低秩层实现
# ============================================================================

class ALRDLinearFP16(nn.Module):
    """FP16 版本的低秩层"""
    
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.rank = rank
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.quantizer = Quantizer(n_bits=4)
    
    def quantize_latent(self, x):
        return self.quantizer(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.BLinear(x)
        latent = self.quantizer(latent)
        return self.ALinear(latent)


class ALRDLinearINT8(nn.Module):
    """INT8 版本的低秩层 (使用 bitsandbytes)"""
    
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.has_bias = bias
        
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.ALinear_bnb = None  # 延迟初始化
        self.quantizer = Quantizer(n_bits=4)
        self._prepared = False
    
    def quantize_latent(self, x):
        return self.quantizer(x)
    
    def prepare_int8(self):
        """准备 INT8 权重"""
        if self._prepared:
            return
        
        if BNB_AVAILABLE:
            device = self.ALinear.weight.device
            self.ALinear_bnb = create_bnb_linear(
                self.rank,
                self.out_features,
                self.ALinear.weight.data,
                self.ALinear.bias if self.has_bias else None,
                device=str(device)
            )
        self._prepared = True
    
    def forward_int8(self, latent: torch.Tensor) -> torch.Tensor:
        """INT8 版本的重建"""
        if not self._prepared:
            self.prepare_int8()
        
        if self.ALinear_bnb is not None:
            return self.ALinear_bnb(latent)
        else:
            # Fallback to FP16
            return self.ALinear(latent)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.BLinear(x)
        latent = self.quantizer(latent)
        return self.forward_int8(latent)


# ============================================================================
# 自定义 Attention (FlashAttention2)
# ============================================================================

class CustomAttentionFP16(LlamaFlashAttention2):
    """FP16 版本的 FlashAttention2"""
    
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._flash_attn_uses_top_left_mask = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        
        # 低秩 K, V
        key_states = self.k_proj.BLinear(hidden_states)
        key_states = self.k_proj.quantize_latent(key_states)
        value_states = self.v_proj.BLinear(hidden_states)
        value_states = self.v_proj.quantize_latent(value_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 更新 KV Cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        
        # 重建 K, V
        key_states = self.k_proj.ALinear(key_states)
        value_states = self.v_proj.ALinear(value_states)
        
        _, k_len, _ = key_states.size()
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        cos, sin = position_embeddings
        if q_len > 1:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos[:, -1:], sin[:, -1:])
            key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        
        # Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        dropout_rate = self.attention_dropout if self.training else 0.0
        
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            target_dtype = torch.float16
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


class CustomAttentionINT8(LlamaFlashAttention2):
    """INT8 版本的 FlashAttention2 (Value 重建使用 INT8)"""
    
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._flash_attn_uses_top_left_mask = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        
        # 低秩 K, V
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
        cos, sin = position_embeddings
        if q_len > 1:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos[:, -1:], sin[:, -1:])
            key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        
        # Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        dropout_rate = self.attention_dropout if self.training else 0.0
        
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            target_dtype = torch.float16
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
# 模型创建和修改
# ============================================================================

def create_model(config: LlamaConfig, device: str = "cuda", dtype = torch.float16):
    """创建随机初始化的模型"""
    model = LlamaForCausalLM(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def replace_with_lowrank_fp16(model: LlamaForCausalLM, rank: int):
    """替换 k_proj, v_proj 为低秩 FP16 版本"""
    config = model.config
    
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        
        # 创建新的 attention
        new_attn = CustomAttentionFP16(config, layer_idx)
        
        # 复制权重
        new_attn.q_proj = old_attn.q_proj
        new_attn.o_proj = old_attn.o_proj
        
        # 创建低秩 k_proj, v_proj
        in_features = config.hidden_size
        k_out = config.num_key_value_heads * config.hidden_size // config.num_attention_heads
        v_out = k_out
        
        new_attn.k_proj = ALRDLinearFP16(in_features, k_out, rank, bias=False)
        new_attn.v_proj = ALRDLinearFP16(in_features, v_out, rank, bias=False)
        
        # 移动到正确设备
        device = old_attn.q_proj.weight.device
        dtype = old_attn.q_proj.weight.dtype
        new_attn.k_proj = new_attn.k_proj.to(device=device, dtype=dtype)
        new_attn.v_proj = new_attn.v_proj.to(device=device, dtype=dtype)
        
        layer.self_attn = new_attn
    
    return model


def replace_with_lowrank_int8(model: LlamaForCausalLM, rank: int):
    """替换 k_proj, v_proj 为低秩 INT8 版本"""
    config = model.config
    
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        
        # 创建新的 attention
        new_attn = CustomAttentionINT8(config, layer_idx)
        
        # 复制权重
        new_attn.q_proj = old_attn.q_proj
        new_attn.o_proj = old_attn.o_proj
        
        # 创建低秩 k_proj (FP16), v_proj (INT8)
        in_features = config.hidden_size
        k_out = config.num_key_value_heads * config.hidden_size // config.num_attention_heads
        v_out = k_out
        
        new_attn.k_proj = ALRDLinearFP16(in_features, k_out, rank, bias=False)
        new_attn.v_proj = ALRDLinearINT8(in_features, v_out, rank, bias=False)
        
        # 移动到正确设备
        device = old_attn.q_proj.weight.device
        dtype = old_attn.q_proj.weight.dtype
        new_attn.k_proj = new_attn.k_proj.to(device=device, dtype=dtype)
        new_attn.v_proj = new_attn.v_proj.to(device=device, dtype=dtype)
        
        layer.self_attn = new_attn
    
    # 准备 INT8 权重
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'v_proj') and isinstance(layer.self_attn.v_proj, ALRDLinearINT8):
            layer.self_attn.v_proj.prepare_int8()
    
    return model


# ============================================================================
# Benchmark 函数
# ============================================================================

def benchmark_prefill(model, input_ids, num_runs=10, warmup=3):
    """测试 prefill 阶段的性能"""
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, use_cache=True)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(input_ids, use_cache=True)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    return sum(times) / len(times)


def benchmark_decode(model, input_ids, num_new_tokens=32, num_runs=5, warmup=2):
    """测试 decode 阶段的性能"""
    
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            past_key_values = DynamicCache()
            outputs = model(input_ids, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            
            next_token = torch.randint(0, model.config.vocab_size, (batch_size, 1), device=device)
            for _ in range(num_new_tokens):
                outputs = model(next_token, use_cache=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                next_token = torch.randint(0, model.config.vocab_size, (batch_size, 1), device=device)
    
    # Benchmark
    decode_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            past_key_values = DynamicCache()
            outputs = model(input_ids, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            
            next_token = torch.randint(0, model.config.vocab_size, (batch_size, 1), device=device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            for _ in range(num_new_tokens):
                outputs = model(next_token, use_cache=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                next_token = torch.randint(0, model.config.vocab_size, (batch_size, 1), device=device)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            decode_times.append((end - start) * 1000 / num_new_tokens)  # ms per token
    
    return sum(decode_times) / len(decode_times)


def get_memory_usage():
    """获取当前 GPU 内存使用"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3  # GB
    return 0


def run_benchmark(config_name: str, config: LlamaConfig, rank: int, 
                  batch_size: int, seq_len: int, device: str = "cuda"):
    """运行完整的 benchmark"""
    
    print(f"\n{'=' * 70}")
    print(f"Benchmark: {config_name}")
    print(f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, heads={config.num_attention_heads}")
    print(f"Batch={batch_size}, SeqLen={seq_len}, Rank={rank}")
    print(f"{'=' * 70}")
    
    # 创建输入
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    results = {}
    
    # ===== FP16 模型 =====
    print("\n[FP16 Model]")
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    
    model_fp16 = create_model(config, device=device)
    model_fp16 = replace_with_lowrank_fp16(model_fp16, rank)
    
    fp16_prefill = benchmark_prefill(model_fp16, input_ids)
    fp16_decode = benchmark_decode(model_fp16, input_ids)
    fp16_memory = get_memory_usage()
    
    print(f"  Prefill: {fp16_prefill:.2f} ms")
    print(f"  Decode: {fp16_decode:.2f} ms/token")
    print(f"  Peak Memory: {fp16_memory:.2f} GB")
    
    results['fp16'] = {
        'prefill': fp16_prefill,
        'decode': fp16_decode,
        'memory': fp16_memory,
    }
    
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== INT8 模型 =====
    print("\n[INT8 Model (bitsandbytes)]")
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    
    model_int8 = create_model(config, device=device)
    model_int8 = replace_with_lowrank_int8(model_int8, rank)
    
    int8_prefill = benchmark_prefill(model_int8, input_ids)
    int8_decode = benchmark_decode(model_int8, input_ids)
    int8_memory = get_memory_usage()
    
    print(f"  Prefill: {int8_prefill:.2f} ms")
    print(f"  Decode: {int8_decode:.2f} ms/token")
    print(f"  Peak Memory: {int8_memory:.2f} GB")
    
    results['int8'] = {
        'prefill': int8_prefill,
        'decode': int8_decode,
        'memory': int8_memory,
    }
    
    del model_int8
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== 对比 =====
    print("\n[Comparison]")
    prefill_speedup = fp16_prefill / int8_prefill
    decode_speedup = fp16_decode / int8_decode
    memory_saving = (fp16_memory - int8_memory) / fp16_memory * 100 if fp16_memory > 0 else 0
    
    print(f"  Prefill Speedup: {prefill_speedup:.2f}x")
    print(f"  Decode Speedup: {decode_speedup:.2f}x")
    print(f"  Memory Saving: {memory_saving:.1f}%")
    
    results['speedup'] = {
        'prefill': prefill_speedup,
        'decode': decode_speedup,
        'memory_saving': memory_saving,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='1b', choices=['tiny', '1b', '7b'],
                        help='Model size to benchmark')
    parser.add_argument('--rank', type=int, default=256, help='Low rank dimension')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    check_backends()
    
    # 定义模型配置
    configs = {
        'tiny': LlamaConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            vocab_size=32000,
            max_position_embeddings=4096,
            use_cache=True,
        ),
        '1b': LlamaConfig(
            hidden_size=2048,
            intermediate_size=5632,
            num_hidden_layers=22,
            num_attention_heads=32,
            num_key_value_heads=4,
            vocab_size=32000,
            max_position_embeddings=4096,
            use_cache=True,
        ),
        '7b': LlamaConfig(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=32000,
            max_position_embeddings=4096,
            use_cache=True,
        ),
    }
    
    config = configs[args.model_size]
    
    print("\n" + "#" * 70)
    print(f"# BENCHMARKING {args.model_size.upper()} MODEL")
    print("#" * 70)
    
    # 运行多个配置
    test_configs = [
        (args.batch_size, args.seq_len),
        (1, 128),
        (1, 512),
        (1, 1024),
        (4, 256),
    ]
    
    all_results = []
    for batch_size, seq_len in test_configs:
        try:
            results = run_benchmark(
                config_name=f"{args.model_size}",
                config=config,
                rank=args.rank,
                batch_size=batch_size,
                seq_len=seq_len,
                device=args.device,
            )
            results['batch_size'] = batch_size
            results['seq_len'] = seq_len
            all_results.append(results)
        except Exception as e:
            print(f"Error with batch={batch_size}, seq_len={seq_len}: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Batch':>6} {'SeqLen':>8} {'FP16 Pre':>10} {'INT8 Pre':>10} {'Speedup':>10}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['batch_size']:>6} {r['seq_len']:>8} "
              f"{r['fp16']['prefill']:>10.2f} {r['int8']['prefill']:>10.2f} "
              f"{r['speedup']['prefill']:>10.2f}x")
    
    print(f"\n{'Batch':>6} {'SeqLen':>8} {'FP16 Dec':>10} {'INT8 Dec':>10} {'Speedup':>10}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['batch_size']:>6} {r['seq_len']:>8} "
              f"{r['fp16']['decode']:>10.2f} {r['int8']['decode']:>10.2f} "
              f"{r['speedup']['decode']:>10.2f}x")
    
    # 结论
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    avg_prefill = sum(r['speedup']['prefill'] for r in all_results) / len(all_results)
    avg_decode = sum(r['speedup']['decode'] for r in all_results) / len(all_results)
    
    print(f"\nAverage Prefill Speedup: {avg_prefill:.2f}x")
    print(f"Average Decode Speedup: {avg_decode:.2f}x")
    
    if avg_prefill < 1.0 and avg_decode < 1.0:
        print("\n⚠️  INT8 is slower than FP16 in this configuration.")
        print("   Consider:")
        print("   1. Using INT8 only for memory savings (KV cache compression)")
        print("   2. Using larger matrices where INT8 has advantage")
        print("   3. Sticking with FP16 for compute")
    elif avg_decode > 1.0:
        print("\n✓ INT8 provides speedup for decode phase!")
    else:
        print("\n⚠️  Mixed results - INT8 may help in some scenarios")


if __name__ == "__main__":
    main()
