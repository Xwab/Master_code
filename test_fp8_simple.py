"""
测试 torch._scaled_mm 的正确用法
"""

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

device = 'cuda'

# 测试不同的矩阵布局
M, K, N = 64, 128, 256

print(f"\n测试 _scaled_mm: ({M}, {K}) @ ({K}, {N})")
print("=" * 60)

# 方法 1: A row-major, B 作为转置创建
print("\n方法 1: B 作为 (N, K) 创建，然后转置")
try:
    A = torch.randn(M, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    # 创建 B 为 (N, K)，然后转置得到 (K, N) column-major
    B_nk = torch.randn(N, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B = B_nk.t()  # (K, N) column-major
    
    print(f"  A shape: {A.shape}, strides: {A.stride()}, contiguous: {A.is_contiguous()}")
    print(f"  B shape: {B.shape}, strides: {B.stride()}, contiguous: {B.is_contiguous()}")
    
    scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
    scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
    print(f"  ✓ 成功! C shape: {C.shape}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 方法 2: 使用 .t().contiguous().t()
print("\n方法 2: B.t().contiguous().t()")
try:
    A = torch.randn(M, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B_orig = torch.randn(K, N, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B = B_orig.t().contiguous().t()
    
    print(f"  A shape: {A.shape}, strides: {A.stride()}, contiguous: {A.is_contiguous()}")
    print(f"  B shape: {B.shape}, strides: {B.stride()}, contiguous: {B.is_contiguous()}")
    
    scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
    scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
    print(f"  ✓ 成功! C shape: {C.shape}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 方法 3: 直接检查 PyTorch 期望的格式
print("\n方法 3: 检查 strides 要求")
try:
    # Row-major: strides = (K, 1)
    A = torch.randn(M, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    
    # Column-major: strides = (1, K) for shape (K, N)
    # 这意味着 B[i, j] = data[i + j*K]
    # 可以通过创建 (N, K) 然后转置得到
    B_storage = torch.randn(N, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B = B_storage.t()  # 现在 B 是 (K, N)，strides = (1, K)
    
    print(f"  A shape: {A.shape}, strides: {A.stride()}")
    print(f"  B shape: {B.shape}, strides: {B.stride()}")
    print(f"  B.t() is contiguous: {B.t().is_contiguous()}")
    
    scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
    scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
    print(f"  ✓ 成功! C shape: {C.shape}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 方法 4: 尝试不同的 scale 格式
print("\n方法 4: 尝试 scale 作为 Tensor vs scalar")
try:
    A = torch.randn(M, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B_storage = torch.randn(N, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B = B_storage.t()
    
    # 尝试不同的 scale 格式
    scale_a = torch.ones((), dtype=torch.float32, device=device)
    scale_b = torch.ones((), dtype=torch.float32, device=device)
    
    print(f"  scale_a: {scale_a.shape}, {scale_a.dtype}")
    print(f"  scale_b: {scale_b.shape}, {scale_b.dtype}")
    
    C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
    print(f"  ✓ 成功! C shape: {C.shape}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 方法 5: 检查 PyTorch 版本特定的 API
print("\n方法 5: 检查 _scaled_mm 签名")
try:
    import inspect
    sig = inspect.signature(torch._scaled_mm)
    print(f"  签名: {sig}")
except:
    print("  无法获取签名")

# 方法 6: 使用 torch.nn.functional 的方式
print("\n方法 6: 尝试其他参数")
try:
    A = torch.randn(M, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B_storage = torch.randn(N, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B = B_storage.t()
    
    # 不传 scale，看是否有默认值
    C = torch._scaled_mm(A, B, out_dtype=torch.float16)
    print(f"  ✓ 成功 (无 scale)! C shape: {C.shape}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 方法 7: 检查是否需要特定的对齐
print("\n方法 7: 使用对齐的维度 (16的倍数)")
try:
    M2, K2, N2 = 64, 128, 256  # 都是16的倍数
    
    A = torch.randn(M2, K2, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B_storage = torch.randn(N2, K2, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B = B_storage.t()
    
    scale_a = torch.tensor(1.0, device=device)
    scale_b = torch.tensor(1.0, device=device)
    
    C = torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
    print(f"  ✓ 成功! C shape: {C.shape}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
