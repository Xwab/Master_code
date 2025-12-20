"""
Diagnose INT8 Performance Issues
"""

import torch
import time

print("=" * 60)
print("INT8 Performance Diagnosis")
print("=" * 60)

# 1. Environment
print("\n[1] Environment")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    
    # Ada Lovelace (L20, RTX 4090) = 8.9
    # Ampere (A100) = 8.0
    # Hopper (H100) = 9.0
    if cap[0] >= 8:
        print("✓ GPU supports INT8 Tensor Cores")
    else:
        print("✗ GPU may not have efficient INT8 support")

# 2. Check _int_mm behavior
print("\n[2] Testing torch._int_mm")

device = 'cuda'
M, K, N = 256, 256, 256

A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)

try:
    C = torch._int_mm(A, B)
    print(f"✓ _int_mm works")
    print(f"  Input dtypes: A={A.dtype}, B={B.dtype}")
    print(f"  Output dtype: {C.dtype}")
    print(f"  Output shape: {C.shape}")
except Exception as e:
    print(f"✗ _int_mm failed: {e}")

# 3. Check if using Tensor Cores
print("\n[3] Checking Tensor Core utilization")
print("Running profiler...")

# Use torch profiler to check
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(10):
        C = torch._int_mm(A, B)
    torch.cuda.synchronize()

# Print kernel names
events = prof.key_averages()
for evt in events:
    if 'int' in evt.key.lower() or 'gemm' in evt.key.lower() or 'cutlass' in evt.key.lower():
        print(f"  Kernel: {evt.key}")
        print(f"    CUDA time: {evt.cuda_time_total / 1000:.3f} ms")

# 4. Alternative: Use torch.compile
print("\n[4] Testing with torch.compile")

@torch.compile
def int8_matmul_compiled(a, b):
    return torch._int_mm(a, b)

# Warmup
for _ in range(5):
    _ = int8_matmul_compiled(A, B)
torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(100):
    _ = int8_matmul_compiled(A, B)
torch.cuda.synchronize()
compiled_time = (time.perf_counter() - start) / 100 * 1000

# Without compile
start = time.perf_counter()
for _ in range(100):
    _ = torch._int_mm(A, B)
torch.cuda.synchronize()
raw_time = (time.perf_counter() - start) / 100 * 1000

print(f"  Without compile: {raw_time:.4f} ms")
print(f"  With compile:    {compiled_time:.4f} ms")

# 5. Compare with scaled_mm (if available, PyTorch 2.2+)
print("\n[5] Testing torch._scaled_mm (if available)")
if hasattr(torch, '_scaled_mm'):
    try:
        # scaled_mm is newer and better optimized
        A_fp8 = A.to(torch.float8_e4m3fn) if hasattr(torch, 'float8_e4m3fn') else A.half()
        B_fp8 = B.to(torch.float8_e4m3fn) if hasattr(torch, 'float8_e4m3fn') else B.half()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        # C = torch._scaled_mm(A_fp8, B_fp8, scale_a, scale_b)
        print("  torch._scaled_mm available (for FP8)")
    except Exception as e:
        print(f"  scaled_mm error: {e}")
else:
    print("  torch._scaled_mm not available (need PyTorch 2.2+)")

# 6. Recommend alternatives
print("\n[6] Alternative: Use bitsandbytes or CUTLASS")
print("""
If torch._int_mm is slow, consider:

1. bitsandbytes library:
   pip install bitsandbytes
   
   import bitsandbytes as bnb
   # Use bnb.matmul_4bit or bnb.nn.Linear8bitLt

2. CUTLASS (via torch.compile):
   @torch.compile(mode="max-autotune")
   def fast_int8_mm(a, b):
       return torch._int_mm(a, b)

3. TensorRT-LLM or vLLM:
   These frameworks have highly optimized INT8 kernels

4. Use FP16 with weight-only quantization:
   - Keep activations in FP16
   - Only quantize weights
   - Use torch.matmul with dequantized weights
""")

# 7. Final benchmark comparison
print("\n[7] Final Benchmark: Different Approaches")

M, K, N = 512, 256, 1024
num_runs = 100

A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)

A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
B_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)

# Warmup
for _ in range(10):
    _ = torch.matmul(A_fp16, B_fp16)
    _ = torch._int_mm(A_int8, B_int8)
torch.cuda.synchronize()

# FP16
start = time.perf_counter()
for _ in range(num_runs):
    C = torch.matmul(A_fp16, B_fp16)
torch.cuda.synchronize()
fp16_time = (time.perf_counter() - start) / num_runs * 1000

# INT8
start = time.perf_counter()
for _ in range(num_runs):
    C = torch._int_mm(A_int8, B_int8)
torch.cuda.synchronize()
int8_time = (time.perf_counter() - start) / num_runs * 1000

# Weight-only quant (dequant weight, FP16 matmul)
B_scale = torch.ones(1, device=device)
start = time.perf_counter()
for _ in range(num_runs):
    B_dequant = B_int8.half() * B_scale
    C = torch.matmul(A_fp16, B_dequant)
torch.cuda.synchronize()
woq_time = (time.perf_counter() - start) / num_runs * 1000

print(f"  Matrix: ({M}, {K}) @ ({K}, {N})")
print(f"  FP16 matmul:         {fp16_time:.4f} ms (baseline)")
print(f"  INT8 _int_mm:        {int8_time:.4f} ms (speedup: {fp16_time/int8_time:.2f}x)")
print(f"  Weight-only quant:   {woq_time:.4f} ms (speedup: {fp16_time/woq_time:.2f}x)")

print("\n" + "=" * 60)
print("Diagnosis Complete")
print("=" * 60)
