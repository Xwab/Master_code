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
        
        # 创建测试数据
        M, K, N = 1024, 256, 1024
        
        a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
        b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        
        # 测试 INT8 matmul
        if device == 'cuda':
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
            
            # FP16
            a_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
            b_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_iter):
                _ = torch.matmul(a_fp16, b_fp16)
            torch.cuda.synchronize()
            fp16_time = (time.perf_counter() - start) / n_iter * 1000
            
            # INT8
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_iter):
                _ = torch._int_mm(a, b)
            torch.cuda.synchronize()
            int8_time = (time.perf_counter() - start) / n_iter * 1000
            
            print(f"FP16 matmul: {fp16_time:.4f} ms")
            print(f"INT8 matmul: {int8_time:.4f} ms")
            print(f"Speedup: {fp16_time/int8_time:.2f}x")
            
            return True
        else:
            print("⚠️  CUDA 不可用，无法测试 torch._int_mm")
            return False
            
    except Exception as e:
        print(f"❌ torch._int_mm 失败: {e}")
        return False


def test_fallback():
    """测试 INT32 fallback"""
    print("\n" + "=" * 70)
    print("INT32 Fallback Test (No speedup expected)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    M, K, N = 1024, 256, 1024
    n_iter = 100
    
    # INT8 -> INT32 fallback
    a_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    b_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.matmul(a_int8.int(), b_int8.int())  # 转为 INT32
    if device == 'cuda':
        torch.cuda.synchronize()
    int32_time = (time.perf_counter() - start) / n_iter * 1000
    
    # FP16
    a_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    b_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.matmul(a_fp16, b_fp16)
    if device == 'cuda':
        torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / n_iter * 1000
    
    print(f"FP16 matmul:          {fp16_time:.4f} ms")
    print(f"INT32 fallback:       {int32_time:.4f} ms")
    print(f"Ratio (should be ~1): {fp16_time/int32_time:.2f}x")
    print("\n⚠️  INT32 fallback 不会有速度提升!")


if __name__ == "__main__":
    has_int8 = test_int8_support()
    test_fallback()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if has_int8:
        print("✅ 可以使用真正的 INT8 矩阵乘法 (torch._int_mm)")
    else:
        print("❌ 无法使用真正的 INT8 矩阵乘法")
        print("   将使用 INT32 fallback (无速度提升)")
    print("=" * 70)
