"""
诊断 INT8 后端问题
"""
import torch
import time

print("=" * 70)
print("INT8 Backend Diagnostic")
print("=" * 70)

# 1. 检查环境
print("\n[1] Environment Check")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")

# 2. 检查 torch._int_mm
print("\n[2] torch._int_mm Check")
has_int_mm = hasattr(torch, '_int_mm')
print(f"torch._int_mm exists: {has_int_mm}")

if has_int_mm and torch.cuda.is_available():
    try:
        a = torch.randint(-128, 127, (64, 128), dtype=torch.int8, device='cuda')
        b = torch.randint(-128, 127, (128, 64), dtype=torch.int8, device='cuda')
        c = torch._int_mm(a, b)
        print(f"torch._int_mm works: True")
        print(f"Output dtype: {c.dtype}")
    except Exception as e:
        print(f"torch._int_mm error: {e}")

# 3. 检查 Triton
print("\n[3] Triton Check")
try:
    import triton
    print(f"Triton version: {triton.__version__}")
    import triton.language as tl
    print("Triton language: OK")
except ImportError as e:
    print(f"Triton not installed: {e}")
except Exception as e:
    print(f"Triton error: {e}")

# 4. 性能对比测试
print("\n[4] Performance Test")

if torch.cuda.is_available():
    device = 'cuda'
    dtype = torch.float16
    
    M, K, N = 1024, 256, 1024
    n_iter = 100
    
    # 创建数据
    x = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(N, K, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(x, w.T)
    torch.cuda.synchronize()
    
    # Test 1: FP16 matmul
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.matmul(x, w.T)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / n_iter * 1000
    print(f"FP16 matmul: {fp16_time:.4f} ms")
    
    # Test 2: INT8 with torch._int_mm
    if has_int_mm:
        x_int8 = (x / x.abs().max() * 127).round().clamp(-128, 127).to(torch.int8)
        w_int8 = (w / w.abs().max() * 127).round().clamp(-128, 127).to(torch.int8)
        
        # Warmup
        for _ in range(10):
            _ = torch._int_mm(x_int8, w_int8.T.contiguous())
        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = torch._int_mm(x_int8, w_int8.T.contiguous())
        torch.cuda.synchronize()
        int8_time = (time.perf_counter() - start) / n_iter * 1000
        print(f"INT8 _int_mm: {int8_time:.4f} ms")
        print(f"Speedup: {fp16_time / int8_time:.2f}x")
    
    # Test 3: Fallback (这就是慢的原因!)
    x_int8 = (x / x.abs().max() * 127).round().clamp(-128, 127).to(torch.int8)
    w_int8 = (w / w.abs().max() * 127).round().clamp(-128, 127).to(torch.int8)
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(x_int8.float(), w_int8.T.float())
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.matmul(x_int8.float(), w_int8.T.float())
    torch.cuda.synchronize()
    fallback_time = (time.perf_counter() - start) / n_iter * 1000
    print(f"Fallback (int8->float): {fallback_time:.4f} ms")
    print(f"Fallback vs FP16: {fp16_time / fallback_time:.2f}x  <-- 这就是你看到的 0.09x!")

print("\n" + "=" * 70)
print("Conclusion")
print("=" * 70)
print("""
如果你看到 0.09x 加速比，说明你用的是 Fallback 后端！

Fallback 的问题:
  torch.matmul(x_int8.float(), w_int8.T.float())
  
  这会:
  1. 把 int8 转成 float32 (慢!)
  2. 做 float32 matmul (比 FP16 慢!)
  3. 完全没有 INT8 加速

解决方案:
  1. 确保 torch._int_mm 可用 (PyTorch 2.0+)
  2. 或者安装 Triton: pip install triton
  3. 或者使用其他 INT8 库如 bitsandbytes
""")
