"""
测试 INT8 矩阵乘法支持情况
"""
import torch
import time

def test_int8_support():
    print("=" * 70)
    print("INT8 Matrix Multiplication Support Test")
    print("=" * 70)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # 检查 torch._int_mm
    print(f"\ntorch._int_mm exists: {hasattr(torch, '_int_mm')}")
    
    if not hasattr(torch, '_int_mm'):
        print("\n⚠️  torch._int_mm 不可用!")
        print("需要 PyTorch 2.0+ 且 CUDA 支持")
        return False
    
    # 测试 torch._int_mm
    print("\n" + "-" * 70)
    print("Testing torch._int_mm...")
    print("-" * 70)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device != 'cuda':
            print("⚠️  CUDA 不可用，无法测试 torch._int_mm")
            return False
        
        # 创建测试数据
        M, K, N = 1024, 256, 1024
        
        a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
        b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        
        # 测试 INT8 matmul
        torch.cuda.synchronize()
        start = time.perf_counter()
        c = torch._int_mm(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        print(f"✅ torch._int_mm 成功!")
        print(f"   Shape: ({M}, {K}) @ ({K}, {N}) -> {c.shape}")
        print(f"   Output dtype: {c.dtype}")
        print(f"   Time: {(end-start)*1000:.4f} ms")
        
        # Benchmark: INT8 vs FP16
        print("\n" + "-" * 70)
        print("Benchmark: INT8 vs FP16")
        print("-" * 70)
        
        n_iter = 100
        
        # Warmup
        for _ in range(10):
            _ = torch._int_mm(a, b)
        torch.cuda.synchronize()
        
        # INT8
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = torch._int_mm(a, b)
        torch.cuda.synchronize()
        int8_time = (time.perf_counter() - start) / n_iter * 1000
        
        # FP16
        a_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
        b_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / n_iter * 1000
        
        print(f"FP16 matmul: {fp16_time:.4f} ms")
        print(f"INT8 matmul: {int8_time:.4f} ms")
        print(f"Speedup: {fp16_time/int8_time:.2f}x")
        
        return True
            
    except Exception as e:
        print(f"❌ torch._int_mm 失败: {e}")
        return False


def test_cpu_fallback():
    """测试 CPU 上的 fallback (使用 float32 模拟)"""
    print("\n" + "=" * 70)
    print("CPU Fallback Test (float32 simulation)")
    print("=" * 70)
    
    M, K, N = 512, 128, 512
    n_iter = 50
    
    # 在 CPU 上，我们只能用 float 来模拟
    a_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    b_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8)
    
    # 模拟 INT8 量化 + 反量化的过程
    a_scale = torch.randn(M, 1).abs() / 127
    b_scale = torch.randn(1, N).abs() / 127
    
    # 方法1: 先转 float 再乘
    start = time.perf_counter()
    for _ in range(n_iter):
        out = torch.matmul(a_int8.float(), b_int8.float()) * a_scale * b_scale
    float_time = (time.perf_counter() - start) / n_iter * 1000
    
    # 方法2: 直接 FP32
    a_fp32 = torch.randn(M, K, dtype=torch.float32)
    b_fp32 = torch.randn(K, N, dtype=torch.float32)
    
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.matmul(a_fp32, b_fp32)
    fp32_time = (time.perf_counter() - start) / n_iter * 1000
    
    print(f"FP32 matmul:           {fp32_time:.4f} ms")
    print(f"INT8->FP32 simulation: {float_time:.4f} ms")
    print(f"\n⚠️  CPU 上无法获得 INT8 加速!")


def test_cuda_supported_dtypes():
    """测试 CUDA 支持的 matmul 数据类型"""
    print("\n" + "=" * 70)
    print("CUDA Supported Matmul Data Types")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = 'cuda'
    M, K, N = 256, 128, 256
    
    dtypes_to_test = [
        ('float32', torch.float32),
        ('float16', torch.float16),
        ('bfloat16', torch.bfloat16),
        ('int8 (via _int_mm)', 'int8'),
        ('int32', torch.int32),
        ('int64', torch.int64),
    ]
    
    for name, dtype in dtypes_to_test:
        try:
            if dtype == 'int8':
                a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
                b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
                _ = torch._int_mm(a, b)
            else:
                if dtype in [torch.int32, torch.int64]:
                    a = torch.randint(-100, 100, (M, K), dtype=dtype, device=device)
                    b = torch.randint(-100, 100, (K, N), dtype=dtype, device=device)
                else:
                    a = torch.randn(M, K, dtype=dtype, device=device)
                    b = torch.randn(K, N, dtype=dtype, device=device)
                _ = torch.matmul(a, b)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {str(e)[:50]}...")


if __name__ == "__main__":
    has_int8 = test_int8_support()
    test_cuda_supported_dtypes()
    test_cpu_fallback()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if has_int8:
        print("✅ 可以使用真正的 INT8 矩阵乘法 (torch._int_mm)")
        print("   这会带来实际的速度提升!")
    else:
        print("❌ 无法使用真正的 INT8 矩阵乘法")
        print("   可选方案:")
        print("   1. 升级 PyTorch 到 2.0+")
        print("   2. 使用 bitsandbytes 库")
        print("   3. 使用 TensorRT")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Important Notes")
    print("=" * 70)
    print("""
CUDA matmul 支持的数据类型:
  ✅ float32, float16, bfloat16
  ✅ int8 (仅通过 torch._int_mm)
  ❌ int32, int64 (不支持!)

所以我之前的 fallback 代码:
  torch.matmul(a_int8.int(), b_int8.int())  # ❌ 会报错!
  
正确的 fallback 应该是:
  torch.matmul(a_int8.float(), b_int8.float())  # ✅ 转为 float
""")
    print("=" * 70)
