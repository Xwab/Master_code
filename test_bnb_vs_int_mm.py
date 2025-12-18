"""
bitsandbytes vs torch._int_mm 性能对比

测试不同矩阵大小下两种 INT8 后端的性能
"""

import torch
import time
from typing import Tuple


def check_backends():
    """检查可用的后端"""
    print("=" * 60)
    print("Backend Detection")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # torch._int_mm
    has_int_mm = hasattr(torch, '_int_mm')
    print(f"torch._int_mm: {has_int_mm}")
    
    # bitsandbytes
    has_bnb = False
    bnb = None
    try:
        import bitsandbytes as bnb_module
        bnb = bnb_module
        has_bnb = True
        print(f"bitsandbytes: True (v{bnb.__version__})")
    except ImportError:
        print("bitsandbytes: False")
    
    print("=" * 60)
    return has_int_mm, has_bnb, bnb


def create_bnb_linear(in_features: int, out_features: int, weight: torch.Tensor, device: str = "cuda"):
    """创建 bitsandbytes 的 8-bit Linear 层"""
    import bitsandbytes as bnb
    
    # 创建 Linear8bitLt 层
    linear = bnb.nn.Linear8bitLt(
        in_features, 
        out_features, 
        bias=False,
        has_fp16_weights=False,
        threshold=0.0,  # 不使用混合精度阈值
    )
    
    # 复制权重并移动到 CUDA（触发量化）
    linear.weight = bnb.nn.Int8Params(
        weight.to(device).contiguous(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    
    return linear.to(device)


@torch.no_grad()
def quantize_int8_symmetric(x: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """对称 INT8 量化"""
    amax = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def benchmark_fn(fn, warmup=20, iterations=100):
    """测试执行时间"""
    for _ in range(warmup):
        fn()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        fn()
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) * 1000 / iterations


def test_performance(M: int, K: int, N: int, has_int_mm: bool, has_bnb: bool, bnb):
    """测试不同后端的性能"""
    print(f"\nMatrix size: ({M}, {K}) x ({K}, {N})")
    print("-" * 50)
    
    device = "cuda"
    dtype = torch.float16
    
    # 创建输入
    x = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(N, K, device=device, dtype=dtype)  # 权重是 (out, in)
    
    results = {}
    
    # 1. FP16 baseline (使用 nn.Linear)
    fp16_linear = torch.nn.Linear(K, N, bias=False).to(device, dtype)
    fp16_linear.weight.data.copy_(w)
    
    def fp16_matmul():
        return fp16_linear(x)
    
    fp16_time = benchmark_fn(fp16_matmul)
    results['FP16'] = fp16_time
    print(f"FP16 Linear: {fp16_time:.4f} ms")
    
    # 2. bitsandbytes Linear8bitLt
    if has_bnb:
        try:
            bnb_linear = create_bnb_linear(K, N, w, device)
            
            # warmup 让 bitsandbytes 完成内部量化
            with torch.no_grad():
                _ = bnb_linear(x)
            
            def bnb_matmul():
                return bnb_linear(x)
            
            bnb_time = benchmark_fn(bnb_matmul)
            results['bnb'] = bnb_time
            speedup = fp16_time / bnb_time
            print(f"bitsandbytes: {bnb_time:.4f} ms (speedup: {speedup:.2f}x)")
        except Exception as e:
            print(f"bitsandbytes: Error - {e}")
    
    # 3. torch._int_mm
    if has_int_mm:
        x_int8, x_scale = quantize_int8_symmetric(x.float(), dim=-1)
        w_int8, w_scale = quantize_int8_symmetric(w.float(), dim=-1)
        
        # 需要确保形状正确
        x_int8 = x_int8.contiguous()
        w_T_int8 = w_int8.T.contiguous()
        
        def int_mm_matmul():
            out = torch._int_mm(x_int8, w_T_int8)
            return out.float() * x_scale * w_scale.T
        
        try:
            int_mm_time = benchmark_fn(int_mm_matmul)
            results['int_mm'] = int_mm_time
            speedup = fp16_time / int_mm_time
            print(f"torch._int_mm: {int_mm_time:.4f} ms (speedup: {speedup:.2f}x)")
        except Exception as e:
            print(f"torch._int_mm: Error - {e}")
    
    # 4. Fallback (float matmul with int8 values)
    x_int8, x_scale = quantize_int8_symmetric(x.float(), dim=-1)
    w_int8, w_scale = quantize_int8_symmetric(w.float(), dim=-1)
    
    def fallback_matmul():
        out = torch.matmul(x_int8.float(), w_int8.T.float())
        return out * x_scale * w_scale.T
    
    fallback_time = benchmark_fn(fallback_matmul)
    results['fallback'] = fallback_time
    speedup = fp16_time / fallback_time
    print(f"Fallback (f32): {fallback_time:.4f} ms (speedup: {speedup:.2f}x)")
    
    return results


def main():
    has_int_mm, has_bnb, bnb = check_backends()
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    # 测试不同矩阵大小
    test_cases = [
        # (M, K, N) - M: batch*seq, K: in_features, N: out_features
        (128, 256, 1024),      # 小矩阵 (decode, small cache)
        (512, 256, 1024),      # 中等矩阵
        (1024, 256, 1024),     # 中等矩阵 (test_int8_support 用的)
        (2048, 256, 4096),     # 较大矩阵
        (4096, 512, 4096),     # 大矩阵 (prefill, long seq)
        (8192, 512, 4096),     # 更大矩阵
        (16384, 256, 4096),    # 非常大 (batch=8, seq=2048)
    ]
    
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    all_results = []
    for M, K, N in test_cases:
        try:
            results = test_performance(M, K, N, has_int_mm, has_bnb, bnb)
            results['shape'] = (M, K, N)
            all_results.append(results)
        except Exception as e:
            print(f"Error testing ({M}, {K}, {N}): {e}")
    
    # 汇总
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Shape':<25} {'FP16':>10} {'bnb':>10} {'int_mm':>10} {'fallback':>10}")
    print("-" * 65)
    
    for r in all_results:
        M, K, N = r['shape']
        fp16 = r.get('FP16', 0)
        bnb_t = r.get('bnb', 0)
        int_mm = r.get('int_mm', 0)
        fallback = r.get('fallback', 0)
        
        print(f"({M:>5},{K:>4},{N:>4}){' '*4}", end="")
        print(f"{fp16:>10.4f}", end="")
        if bnb_t > 0:
            speedup = fp16 / bnb_t
            print(f"{bnb_t:>7.4f}({speedup:.1f}x)", end="")
        else:
            print(f"{'N/A':>10}", end="")
        if int_mm > 0:
            speedup = fp16 / int_mm
            print(f"{int_mm:>7.4f}({speedup:.1f}x)", end="")
        else:
            print(f"{'N/A':>10}", end="")
        print(f"{fallback:>10.4f}")
    
    # 结论
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    
    # 找出哪个后端在大矩阵时更快
    large_results = [r for r in all_results if r['shape'][0] >= 4096]
    
    if large_results and has_bnb:
        avg_bnb_speedup = sum(r['FP16'] / r.get('bnb', float('inf')) for r in large_results if 'bnb' in r) / len(large_results)
        print(f"\nAverage bitsandbytes speedup (large matrices): {avg_bnb_speedup:.2f}x")
    
    if large_results and has_int_mm:
        avg_int_mm_speedup = sum(r['FP16'] / r.get('int_mm', float('inf')) for r in large_results if 'int_mm' in r) / len(large_results)
        print(f"Average torch._int_mm speedup (large matrices): {avg_int_mm_speedup:.2f}x")
    
    print("\nRecommendation:")
    if has_bnb:
        print("  - bitsandbytes is available and optimized for LLM inference")
        print("  - It automatically handles quantization and may provide better speedup")
    if has_int_mm:
        print("  - torch._int_mm is available as fallback")
    print("  - For small matrices, FP16 may still be faster due to overhead")


if __name__ == "__main__":
    main()
