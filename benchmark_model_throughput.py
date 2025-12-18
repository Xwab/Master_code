"""
INT8 vs FP16 模型吞吐量对比 Benchmark

不需要本地模型，使用随机初始化的权重进行测试
主要测试 Value 重建时使用 INT8 矩阵乘法的效果
"""

import torch
import torch.nn as nn
import time
import argparse
from typing import Optional, Tuple


# ============================================================================
# 检测可用的 INT8 后端
# ============================================================================

def check_int8_backends():
    """检测可用的 INT8 后端"""
    print("=" * 60)
    print("INT8 Backend Detection")
    print("=" * 60)
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # torch._int_mm
    has_int_mm = hasattr(torch, '_int_mm')
    print(f"torch._int_mm: {has_int_mm}")
    
    # bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes: True (version {bnb.__version__})")
        has_bnb = True
    except ImportError:
        print("bitsandbytes: False")
        has_bnb = False
    
    # Triton
    try:
        import triton
        print(f"Triton: True (version {triton.__version__})")
        has_triton = True
    except ImportError:
        print("Triton: False")
        has_triton = False
    
    print("=" * 60)
    return {"int_mm": has_int_mm, "bnb": has_bnb, "triton": has_triton}


# ============================================================================
# 量化工具
# ============================================================================

@torch.no_grad()
def quantize_to_int8_symmetric(x: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """对称 INT8 量化"""
    amax = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


# ============================================================================
# 低秩层实现
# ============================================================================

class ALRDLinearFP16(nn.Module):
    """FP16 版本的低秩层"""
    
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.BLinear(x)
        return self.ALinear(latent)


def create_bnb_linear_for_benchmark(in_features: int, out_features: int, weight: torch.Tensor, 
                                     bias: Optional[torch.Tensor] = None, device: str = "cuda"):
    """创建 bitsandbytes 的 8-bit Linear 层 (用于 benchmark)"""
    import bitsandbytes as bnb
    
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
        linear.bias = torch.nn.Parameter(bias.to(device).contiguous())
    
    return linear.to(device)


class ALRDLinearINT8(nn.Module):
    """INT8 版本的低秩层"""
    
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True, backend: str = "auto"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.has_bias = bias
        self.backend = backend
        
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.ALinear_bnb = None  # bitsandbytes 版本
        
        # 缓存量化权重 (用于 torch._int_mm)
        self.register_buffer('A_int8', None)
        self.register_buffer('A_scale', None)
        self._prepared = False
        
        # 选择后端
        self._select_backend()
    
    def _select_backend(self):
        if self.backend == "auto":
            try:
                import bitsandbytes as bnb
                self.backend = "bnb"
                self._bnb = bnb
            except ImportError:
                if hasattr(torch, '_int_mm'):
                    self.backend = "torch"
                else:
                    self.backend = "fallback"
    
    def prepare_int8_weights(self):
        if self._prepared:
            return
        with torch.no_grad():
            if self.backend == "bnb":
                # 创建 bitsandbytes 8-bit Linear
                device = self.ALinear.weight.device
                self.ALinear_bnb = create_bnb_linear_for_benchmark(
                    self.rank, 
                    self.out_features,
                    self.ALinear.weight.data,
                    self.ALinear.bias if self.has_bias else None,
                    device=str(device)
                )
            else:
                # torch._int_mm 或 fallback
                w = self.ALinear.weight.data.float()
                self.A_int8, self.A_scale = quantize_to_int8_symmetric(w, dim=-1)
            self._prepared = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.BLinear(x)
        return self.forward_int8(latent)
    
    def forward_int8(self, latent: torch.Tensor) -> torch.Tensor:
        """INT8 版本的重建"""
        
        if not self._prepared:
            self.prepare_int8_weights()
        
        if self.backend == "bnb" and self.ALinear_bnb is not None:
            # bitsandbytes Linear8bitLt
            out = self.ALinear_bnb(latent)
            return out.to(latent.dtype)
        
        elif self.backend == "torch" and hasattr(torch, '_int_mm'):
            # torch._int_mm
            bsz, seq_len, in_features = latent.shape
            latent_int8, latent_scale = quantize_to_int8_symmetric(latent.float(), dim=-1)
            
            x_2d = latent_int8.view(-1, in_features).contiguous()
            w_T = self.A_int8.T.contiguous()
            
            out_int32 = torch._int_mm(x_2d, w_T)
            out = out_int32.view(bsz, seq_len, -1).float() * latent_scale * self.A_scale.T
            
            if self.ALinear.bias is not None:
                out = out + self.ALinear.bias
            
            return out.to(latent.dtype)
        
        else:
            # Fallback: FP16
            return self.ALinear(latent)


# ============================================================================
# Benchmark 函数
# ============================================================================

def benchmark_fn(fn, warmup: int = 10, iterations: int = 100) -> float:
    """测试函数执行时间（毫秒）"""
    # Warmup
    for _ in range(warmup):
        fn()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        fn()
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) * 1000 / iterations


def run_benchmark_suite(batch_size: int, seq_len: int, hidden_dim: int, rank: int, 
                        device: str = "cuda", dtype = torch.float16):
    """运行完整的 benchmark 套件"""
    
    print(f"\n{'=' * 70}")
    print(f"Benchmark: batch={batch_size}, seq_len={seq_len}, hidden={hidden_dim}, rank={rank}")
    print(f"{'=' * 70}")
    
    # 创建层
    fp16_layer = ALRDLinearFP16(hidden_dim, hidden_dim, rank).to(device, dtype)
    int8_layer = ALRDLinearINT8(hidden_dim, hidden_dim, rank, backend="auto").to(device, dtype)
    
    # 复制权重
    int8_layer.BLinear.weight.data.copy_(fp16_layer.BLinear.weight.data)
    int8_layer.ALinear.weight.data.copy_(fp16_layer.ALinear.weight.data)
    if fp16_layer.ALinear.bias is not None:
        int8_layer.ALinear.bias.data.copy_(fp16_layer.ALinear.bias.data)
    
    # 准备 INT8 权重
    int8_layer.prepare_int8_weights()
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    
    # 预热
    with torch.no_grad():
        _ = fp16_layer(x)
        _ = int8_layer(x)
    
    # 测试 FP16
    with torch.no_grad():
        fp16_time = benchmark_fn(lambda: fp16_layer(x))
    
    # 测试 INT8
    with torch.no_grad():
        int8_time = benchmark_fn(lambda: int8_layer(x))
    
    # 计算吞吐量
    tokens = batch_size * seq_len
    fp16_throughput = tokens / (fp16_time / 1000)  # tokens/s
    int8_throughput = tokens / (int8_time / 1000)  # tokens/s
    
    speedup = fp16_time / int8_time
    
    print(f"\nBackend used: {int8_layer.backend}")
    print(f"\nLatency:")
    print(f"  FP16: {fp16_time:.4f} ms")
    print(f"  INT8: {int8_time:.4f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    print(f"\nThroughput:")
    print(f"  FP16: {fp16_throughput/1e6:.2f}M tokens/s")
    print(f"  INT8: {int8_throughput/1e6:.2f}M tokens/s")
    
    # 验证精度
    with torch.no_grad():
        fp16_out = fp16_layer(x)
        int8_out = int8_layer(x)
        
        # 相对误差
        rel_error = (fp16_out - int8_out).abs().mean() / fp16_out.abs().mean()
        max_error = (fp16_out - int8_out).abs().max()
    
    print(f"\nAccuracy:")
    print(f"  Mean relative error: {rel_error.item():.6f}")
    print(f"  Max absolute error: {max_error.item():.6f}")
    
    return {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'fp16_time': fp16_time,
        'int8_time': int8_time,
        'speedup': speedup,
        'rel_error': rel_error.item(),
    }


def run_decode_benchmark(batch_size: int, cache_len: int, hidden_dim: int, rank: int,
                         device: str = "cuda", dtype = torch.float16):
    """模拟 Decode 阶段的 benchmark"""
    
    print(f"\n{'=' * 70}")
    print(f"Decode Benchmark: batch={batch_size}, cache_len={cache_len}")
    print(f"{'=' * 70}")
    
    # 创建层 (模拟 Value 重建)
    fp16_layer = ALRDLinearFP16(hidden_dim, hidden_dim, rank).to(device, dtype)
    int8_layer = ALRDLinearINT8(hidden_dim, hidden_dim, rank, backend="auto").to(device, dtype)
    
    # 复制权重
    int8_layer.BLinear.weight.data.copy_(fp16_layer.BLinear.weight.data)
    int8_layer.ALinear.weight.data.copy_(fp16_layer.ALinear.weight.data)
    
    int8_layer.prepare_int8_weights()
    
    # Decode: 单 token 输入，但需要重建整个 cache
    # 模拟从 KV cache 读取后的重建
    latent = torch.randn(batch_size, cache_len, rank, device=device, dtype=dtype)
    
    with torch.no_grad():
        fp16_time = benchmark_fn(lambda: fp16_layer.ALinear(latent))
        int8_time = benchmark_fn(lambda: int8_layer.forward_int8(latent))
    
    speedup = fp16_time / int8_time
    
    print(f"\nLatency (Reconstruction only):")
    print(f"  FP16: {fp16_time:.4f} ms")
    print(f"  INT8: {int8_time:.4f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    # 内存节省估算
    fp16_memory = batch_size * cache_len * rank * 2  # FP16 = 2 bytes
    int8_memory = batch_size * cache_len * rank * 1  # INT8 = 1 byte
    memory_saving = (1 - int8_memory / fp16_memory) * 100
    
    print(f"\nMemory (Latent storage):")
    print(f"  FP16: {fp16_memory / 1e6:.2f} MB")
    print(f"  INT8: {int8_memory / 1e6:.2f} MB")
    print(f"  Saving: {memory_saving:.1f}%")
    
    return {
        'cache_len': cache_len,
        'fp16_time': fp16_time,
        'int8_time': int8_time,
        'speedup': speedup,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=4096, help='Hidden dimension')
    parser.add_argument('--rank', type=int, default=256, help='Low rank dimension')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # 检测后端
    backends = check_int8_backends()
    
    print("\n" + "=" * 70)
    print("BENCHMARK CONFIGURATION")
    print("=" * 70)
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Rank: {args.rank}")
    print(f"Device: {args.device}")
    
    # ========== Prefill Benchmark ==========
    print("\n" + "#" * 70)
    print("# PREFILL BENCHMARKS (Various sequence lengths)")
    print("#" * 70)
    
    prefill_configs = [
        (1, 128),
        (1, 512),
        (1, 1024),
        (1, 2048),
        (4, 512),
        (8, 256),
    ]
    
    prefill_results = []
    for batch_size, seq_len in prefill_configs:
        try:
            result = run_benchmark_suite(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_dim=args.hidden_dim,
                rank=args.rank,
                device=args.device,
            )
            prefill_results.append(result)
        except Exception as e:
            print(f"Error: {e}")
    
    # ========== Decode Benchmark ==========
    print("\n" + "#" * 70)
    print("# DECODE BENCHMARKS (Value reconstruction from cache)")
    print("#" * 70)
    
    decode_configs = [
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (8, 1024),
        (8, 2048),
    ]
    
    decode_results = []
    for batch_size, cache_len in decode_configs:
        try:
            result = run_decode_benchmark(
                batch_size=batch_size,
                cache_len=cache_len,
                hidden_dim=args.hidden_dim,
                rank=args.rank,
                device=args.device,
            )
            decode_results.append(result)
        except Exception as e:
            print(f"Error: {e}")
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nPrefill Results:")
    print(f"{'Batch':>6} {'SeqLen':>8} {'FP16 (ms)':>12} {'INT8 (ms)':>12} {'Speedup':>10}")
    print("-" * 50)
    for r in prefill_results:
        print(f"{r['batch_size']:>6} {r['seq_len']:>8} {r['fp16_time']:>12.4f} {r['int8_time']:>12.4f} {r['speedup']:>10.2f}x")
    
    print("\nDecode Results:")
    print(f"{'CacheLen':>10} {'FP16 (ms)':>12} {'INT8 (ms)':>12} {'Speedup':>10}")
    print("-" * 50)
    for r in decode_results:
        print(f"{r['cache_len']:>10} {r['fp16_time']:>12.4f} {r['int8_time']:>12.4f} {r['speedup']:>10.2f}x")
    
    # 结论
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    avg_prefill_speedup = sum(r['speedup'] for r in prefill_results) / len(prefill_results) if prefill_results else 0
    avg_decode_speedup = sum(r['speedup'] for r in decode_results) / len(decode_results) if decode_results else 0
    
    print(f"\nAverage Prefill Speedup: {avg_prefill_speedup:.2f}x")
    print(f"Average Decode Speedup: {avg_decode_speedup:.2f}x")
    
    if avg_prefill_speedup < 1.0 and avg_decode_speedup < 1.0:
        print("\n⚠️  INT8 is slower than FP16 for both prefill and decode.")
        print("   This is likely due to:")
        print("   1. Small matrix sizes (kernel launch overhead dominates)")
        print("   2. Highly optimized FP16 Tensor Cores on modern GPUs")
        print("   3. Quantization/dequantization overhead")
        print("\n   Consider using INT8 only for memory savings (e.g., KV cache compression)")
    elif avg_decode_speedup > 1.0:
        print("\n✓ INT8 provides speedup for decode (memory-bound operations)")
    else:
        print("\n⚠️  INT8 may only provide memory savings, not compute speedup")


if __name__ == "__main__":
    main()
