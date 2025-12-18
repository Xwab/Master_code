"""
FP8 vs FP16 矩阵乘法性能对比 (v2)

使用 torch.float8 的正确方式
"""

import torch
import torch.nn as nn
import time
import sys


def check_environment():
    """检查环境."""
    print("=" * 70)
    print("环境检查")
    print("=" * 70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if not torch.cuda.is_available():
        return False
    
    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu}")
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    
    if cap < (8, 9):
        print(f"警告: FP8 需要 Ada Lovelace (sm_89+), 当前 sm_{cap[0]}{cap[1]}")
    
    return True


def benchmark(func, num_warmup=20, num_runs=100):
    """Benchmark."""
    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_runs):
        func()
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs


def test_scaled_mm_formats():
    """测试 _scaled_mm 的各种格式."""
    print("\n" + "=" * 70)
    print("测试 torch._scaled_mm 格式")
    print("=" * 70)
    
    device = 'cuda'
    M, K, N = 64, 128, 256
    
    results = []
    
    # 格式 1: A row-major, B 从 (N,K) 转置
    print("\n格式 1: B = randn(N,K).t()")
    try:
        A = torch.randn(M, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B_base = torch.randn(N, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B = B_base.t()  # (K, N) with strides (1, K)
        
        scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
        print(f"  ✓ 成功! A.stride={A.stride()}, B.stride={B.stride()}, C={C.shape}")
        results.append(("格式1", True))
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results.append(("格式1", False))
    
    # 格式 2: 使用 contiguous_format
    print("\n格式 2: A contiguous, B.t() contiguous")
    try:
        A = torch.randn(M, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn).contiguous()
        B_t = torch.randn(N, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn).contiguous()
        B = B_t.t()
        
        scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
        print(f"  ✓ 成功!")
        results.append(("格式2", True))
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results.append(("格式2", False))
    
    # 格式 3: 不传 scale
    print("\n格式 3: 不传 scale 参数")
    try:
        A = torch.randn(M, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B_base = torch.randn(N, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B = B_base.t()
        
        C = torch._scaled_mm(A, B, out_dtype=torch.float16)
        print(f"  ✓ 成功!")
        results.append(("格式3", True))
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results.append(("格式3", False))
    
    # 格式 4: scale 作为 Python float
    print("\n格式 4: scale 作为 Python float")
    try:
        A = torch.randn(M, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B_base = torch.randn(N, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B = B_base.t()
        
        C = torch._scaled_mm(A, B, scale_a=1.0, scale_b=1.0, out_dtype=torch.float16)
        print(f"  ✓ 成功!")
        results.append(("格式4", True))
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results.append(("格式4", False))
    
    # 格式 5: 使用 use_fast_accum
    print("\n格式 5: 带 use_fast_accum 参数")
    try:
        A = torch.randn(M, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B_base = torch.randn(N, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B = B_base.t()
        
        scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16, use_fast_accum=True)
        print(f"  ✓ 成功!")
        results.append(("格式5", True))
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results.append(("格式5", False))
    
    return results


def run_fp8_benchmark_if_available():
    """如果 FP8 可用则运行基准测试."""
    print("\n" + "=" * 70)
    print("FP8 基准测试")
    print("=" * 70)
    
    device = 'cuda'
    
    # 先测试是否能工作
    M, K, N = 64, 128, 256
    
    try:
        A = torch.randn(M, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B_base = torch.randn(N, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
        B = B_base.t()
        
        scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
        print("✓ FP8 _scaled_mm 可用!")
        fp8_available = True
    except Exception as e:
        print(f"✗ FP8 _scaled_mm 不可用: {e}")
        print("\n尝试使用 fake FP8 (转换为 FP16 计算)...")
        fp8_available = False
    
    if not fp8_available:
        print("\n由于 _scaled_mm 不可用，将使用 fake FP8 方式测试量化误差影响")
        run_fake_fp8_benchmark()
        return
    
    # 运行真实 FP8 基准测试
    print("\n运行真实 FP8 基准测试...")
    
    configs = [
        (4096, 4096, 4096, "4K x 4K x 4K"),
        (16384, 4096, 4096, "16K x 4K x 4K"),
        (65536, 4096, 4096, "64K x 4K x 4K"),
        (262144, 4096, 4096, "262K x 4K x 4K"),
        (262144, 256, 1024, "262K x 256 -> 1K"),
        (262144, 256, 4096, "262K x 256 -> 4K"),
    ]
    
    print(f"\n{'配置':<25} {'FP16 (ms)':<12} {'FP8 (ms)':<12} {'Speedup':<10}")
    print("-" * 65)
    
    for M, K, N, desc in configs:
        try:
            # FP16
            A_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
            B_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)
            
            fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 50)
            
            # FP8
            A_fp8 = A_fp16.to(torch.float8_e4m3fn)
            B_base = torch.randn(N, K, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
            B_fp8 = B_base.t()
            
            scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
            scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
            
            def fp8_mm():
                return torch._scaled_mm(A_fp8, B_fp8, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
            
            fp8_time = benchmark(fp8_mm, 10, 50)
            
            speedup = fp16_time / fp8_time
            print(f"{desc:<25} {fp16_time:<12.4f} {fp8_time:<12.4f} {speedup:<10.2f}x")
            
            del A_fp16, B_fp16, A_fp8, B_fp8, B_base
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{desc:<25} OOM")
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")


def run_fake_fp8_benchmark():
    """使用 fake FP8 (dequant to FP16) 的基准测试."""
    print("\n" + "=" * 70)
    print("Fake FP8 基准测试 (量化后转回 FP16 计算)")
    print("=" * 70)
    
    device = 'cuda'
    
    def quantize_fp8_fake(x):
        """Fake FP8 量化: 量化到 FP8 范围，但保持 FP16."""
        max_val = 448.0  # E4M3 max
        scale = x.abs().amax() / max_val
        if scale == 0:
            scale = torch.tensor(1.0, device=x.device)
        x_scaled = x / scale
        # 模拟 FP8 精度损失
        x_fp8 = x_scaled.to(torch.float8_e4m3fn).to(torch.float16)
        return x_fp8 * scale
    
    configs = [
        (4096, 4096, 4096, "4K x 4K x 4K"),
        (16384, 4096, 4096, "16K x 4K x 4K"),
        (65536, 4096, 4096, "64K x 4K x 4K"),
        (262144, 4096, 4096, "262K x 4K x 4K"),
        (262144, 256, 1024, "262K x 256 -> 1K"),
    ]
    
    print(f"\n{'配置':<25} {'FP16 (ms)':<12} {'FakeFP8 (ms)':<12} {'开销':<10}")
    print("-" * 65)
    
    for M, K, N, desc in configs:
        try:
            A_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
            B_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)
            
            # FP16
            fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 50)
            
            # Fake FP8 (量化后转回 FP16)
            def fake_fp8_mm():
                A_q = quantize_fp8_fake(A_fp16)
                B_q = quantize_fp8_fake(B_fp16)
                return torch.matmul(A_q, B_q)
            
            fake_fp8_time = benchmark(fake_fp8_mm, 10, 50)
            
            overhead = fake_fp8_time / fp16_time
            print(f"{desc:<25} {fp16_time:<12.4f} {fake_fp8_time:<12.4f} {overhead:<10.2f}x")
            
            del A_fp16, B_fp16
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{desc:<25} OOM")
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")
    
    print("\n说明: Fake FP8 只测试量化精度损失，不测试实际 FP8 计算速度")


def run_int8_vs_fp16():
    """INT8 vs FP16 对比 (作为参考)."""
    print("\n" + "=" * 70)
    print("INT8 vs FP16 对比 (参考)")
    print("=" * 70)
    
    device = 'cuda'
    
    configs = [
        (4096, 4096, 4096, "4K x 4K x 4K"),
        (16384, 4096, 4096, "16K x 4K x 4K"),
        (65536, 4096, 4096, "64K x 4K x 4K"),
        (262144, 4096, 4096, "262K x 4K x 4K"),
    ]
    
    print(f"\n{'配置':<25} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'INT8/FP16':<10}")
    print("-" * 65)
    
    for M, K, N, desc in configs:
        try:
            A_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
            B_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)
            
            A_int8 = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
            B_int8 = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)
            
            fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 50)
            int8_time = benchmark(lambda: torch._int_mm(A_int8, B_int8), 10, 50)
            
            ratio = int8_time / fp16_time
            print(f"{desc:<25} {fp16_time:<12.4f} {int8_time:<12.4f} {ratio:<10.2f}x")
            
            del A_fp16, B_fp16, A_int8, B_int8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{desc:<25} OOM")
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")
    
    print("\n注意: INT8 输出是 INT32，写带宽是 FP16 的 2x")


def main():
    if not check_environment():
        sys.exit(1)
    
    # 测试 _scaled_mm 格式
    results = test_scaled_mm_formats()
    
    # 检查是否有成功的格式
    successful = [r for r in results if r[1]]
    if successful:
        print(f"\n成功的格式: {[r[0] for r in successful]}")
        run_fp8_benchmark_if_available()
    else:
        print("\n所有 _scaled_mm 格式都失败了")
        print("将使用 fake FP8 测试")
        run_fake_fp8_benchmark()
    
    # INT8 对比
    run_int8_vs_fp16()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
结论:

1. torch._scaled_mm 要求特定的矩阵布局
   - A: row-major (contiguous)
   - B: column-major (B.t() must be contiguous)

2. 如果 _scaled_mm 不工作:
   - 可能是 PyTorch 版本问题
   - 尝试升级到 PyTorch 2.2+
   - 或使用 torchao / transformer_engine

3. 替代方案:
   - torchao: pip install torchao
   - transformer_engine: pip install transformer_engine
   - 这些库提供更稳定的 FP8 支持
    """)


if __name__ == "__main__":
    main()
