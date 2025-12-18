"""
Comprehensive INT8/FP8 Matrix Multiplication Benchmark

Tests all available INT8/FP8 backends on your GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from typing import Callable, Dict, Any

# ============================================================================
# Utility Functions
# ============================================================================

def benchmark(func: Callable, num_warmup: int = 10, num_runs: int = 100) -> float:
    """Benchmark a function and return average time in ms."""
    # Warmup
    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        func()
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_runs * 1000


def check_environment():
    """Check available backends."""
    print("=" * 70)
    print("Environment & Available Backends")
    print("=" * 70)
    
    backends = {}
    
    # Basic info
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {cap[0]}.{cap[1]}")
        
        # Ada Lovelace = 8.9, supports FP8
        if cap >= (8, 9):
            print("✓ FP8 Tensor Cores supported (Ada Lovelace)")
            backends['fp8'] = True
        else:
            backends['fp8'] = False
    
    # torch._int_mm
    backends['torch_int_mm'] = hasattr(torch, '_int_mm')
    print(f"\ntorch._int_mm: {'✓' if backends['torch_int_mm'] else '✗'}")
    
    # torch._scaled_mm (FP8)
    backends['torch_scaled_mm'] = hasattr(torch, '_scaled_mm')
    print(f"torch._scaled_mm (FP8): {'✓' if backends['torch_scaled_mm'] else '✗'}")
    
    # torch.compile
    backends['torch_compile'] = hasattr(torch, 'compile')
    print(f"torch.compile: {'✓' if backends['torch_compile'] else '✗'}")
    
    # Triton
    try:
        import triton
        backends['triton'] = True
        print(f"Triton: ✓ (version {triton.__version__})")
    except ImportError:
        backends['triton'] = False
        print("Triton: ✗ (pip install triton)")
    
    # bitsandbytes
    try:
        import bitsandbytes as bnb
        backends['bitsandbytes'] = True
        print(f"bitsandbytes: ✓ (version {bnb.__version__})")
    except ImportError:
        backends['bitsandbytes'] = False
        print("bitsandbytes: ✗ (pip install bitsandbytes)")
    
    # FBGEMM (usually comes with PyTorch)
    try:
        import torch.ao.quantization
        backends['fbgemm'] = True
        print("FBGEMM (torch.ao): ✓")
    except:
        backends['fbgemm'] = False
        print("FBGEMM: ✗")
    
    return backends


# ============================================================================
# Different INT8/FP8 Implementations
# ============================================================================

def fp16_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Baseline FP16 matmul."""
    return torch.matmul(A, B)


def torch_int_mm(A: torch.Tensor, B: torch.Tensor, 
                 A_scale: torch.Tensor, B_scale: torch.Tensor) -> torch.Tensor:
    """PyTorch native INT8 matmul."""
    A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
    B_int8 = (B / B_scale).round().clamp(-128, 127).to(torch.int8)
    C_int32 = torch._int_mm(A_int8, B_int8)
    return C_int32.half() * A_scale * B_scale


def torch_int_mm_precomputed(A_int8: torch.Tensor, B_int8: torch.Tensor,
                              A_scale: torch.Tensor, B_scale: torch.Tensor) -> torch.Tensor:
    """INT8 matmul with pre-quantized inputs (no quantization overhead)."""
    C_int32 = torch._int_mm(A_int8, B_int8)
    return C_int32.half() * A_scale * B_scale


# ============================================================================
# Triton INT8 Kernel (if available)
# ============================================================================

TRITON_INT8_KERNEL = None

def setup_triton_int8():
    """Setup Triton INT8 kernel."""
    global TRITON_INT8_KERNEL
    
    try:
        import triton
        import triton.language as tl
        
        @triton.jit
        def int8_matmul_kernel(
            A_ptr, B_ptr, C_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        ):
            """Simple INT8 matmul kernel."""
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            
            for k in range(0, K, BLOCK_K):
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0)
                acc += tl.dot(a, b, allow_tf32=False)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            
            c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc, mask=mask)
        
        TRITON_INT8_KERNEL = int8_matmul_kernel
        return True
    except Exception as e:
        print(f"Triton INT8 setup failed: {e}")
        return False


def triton_int8_matmul(A_int8: torch.Tensor, B_int8: torch.Tensor) -> torch.Tensor:
    """Triton INT8 matmul."""
    global TRITON_INT8_KERNEL
    
    M, K = A_int8.shape
    K2, N = B_int8.shape
    assert K == K2
    
    C = torch.empty(M, N, dtype=torch.int32, device=A_int8.device)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    TRITON_INT8_KERNEL[grid](
        A_int8, B_int8, C,
        M, N, K,
        A_int8.stride(0), A_int8.stride(1),
        B_int8.stride(0), B_int8.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return C


# ============================================================================
# torch.compile Optimized
# ============================================================================

@torch.compile(mode="reduce-overhead")
def compiled_fp16_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """torch.compile optimized FP16 matmul."""
    return torch.matmul(A, B)


@torch.compile(mode="max-autotune")
def compiled_int8_matmul(A_int8: torch.Tensor, B_int8: torch.Tensor,
                          A_scale: torch.Tensor, B_scale: torch.Tensor) -> torch.Tensor:
    """torch.compile optimized INT8 matmul."""
    C_int32 = torch._int_mm(A_int8, B_int8)
    return C_int32.half() * A_scale * B_scale


# ============================================================================
# cuBLASLt via torch (implicit)
# ============================================================================

def cublaslt_fp16_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """cuBLASLt FP16 matmul (torch.matmul uses cuBLASLt internally)."""
    # torch.matmul already uses cuBLASLt for FP16 on modern GPUs
    return torch.matmul(A.contiguous(), B.contiguous())


# ============================================================================
# FP8 (Ada Lovelace)
# ============================================================================

def check_fp8_support():
    """Check if FP8 is supported."""
    if not hasattr(torch, 'float8_e4m3fn'):
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap >= (8, 9)  # Ada Lovelace or newer


def fp8_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """FP8 matmul using torch._scaled_mm."""
    # Convert to FP8
    A_scale = A.abs().max() / 448.0  # FP8 E4M3 max value
    B_scale = B.abs().max() / 448.0
    
    A_fp8 = (A / A_scale).to(torch.float8_e4m3fn)
    B_fp8 = (B / B_scale).to(torch.float8_e4m3fn)
    
    # Scaled matmul
    C = torch._scaled_mm(
        A_fp8, B_fp8,
        scale_a=A_scale,
        scale_b=B_scale,
        out_dtype=torch.float16,
    )
    return C


# ============================================================================
# Main Benchmark
# ============================================================================

def run_comprehensive_benchmark(backends: Dict[str, bool]):
    """Run benchmarks for all available backends."""
    print("\n" + "=" * 70)
    print("Comprehensive INT8/FP8 Benchmark")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    # Test configurations
    configs = [
        (128, 256, 1024, "V recon small: 128x256 @ 256x1024"),
        (512, 256, 1024, "V recon med: 512x256 @ 256x1024"),
        (2048, 256, 1024, "V recon long: 2048x256 @ 256x1024"),
        (512, 1024, 1024, "Medium: 512x1024 @ 1024x1024"),
        (2048, 1024, 1024, "Large: 2048x1024 @ 1024x1024"),
        (4096, 4096, 4096, "XL: 4096x4096 @ 4096x4096"),
    ]
    
    for M, K, N, desc in configs:
        print(f"\n{desc}")
        print("-" * 60)
        
        # Create test data
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)
        
        results = {}
        
        # 1. FP16 baseline
        results['FP16'] = benchmark(lambda: fp16_matmul(A, B))
        
        # 2. torch.compile FP16
        if backends.get('torch_compile'):
            try:
                # Warmup compile
                _ = compiled_fp16_matmul(A, B)
                results['FP16 (compiled)'] = benchmark(lambda: compiled_fp16_matmul(A, B))
            except Exception as e:
                results['FP16 (compiled)'] = f"Error: {e}"
        
        # 3. torch._int_mm
        if backends.get('torch_int_mm'):
            A_scale = A.abs().max() / 127
            B_scale = B.abs().max() / 127
            A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
            B_int8 = (B / B_scale).round().clamp(-128, 127).to(torch.int8)
            
            # With quantization overhead
            results['INT8 (_int_mm + quant)'] = benchmark(
                lambda: torch_int_mm(A, B, A_scale, B_scale)
            )
            
            # Without quantization overhead (pre-quantized)
            results['INT8 (_int_mm only)'] = benchmark(
                lambda: torch_int_mm_precomputed(A_int8, B_int8, A_scale, B_scale)
            )
        
        # 4. torch.compile INT8
        if backends.get('torch_compile') and backends.get('torch_int_mm'):
            try:
                A_scale = A.abs().max() / 127
                B_scale = B.abs().max() / 127
                A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
                B_int8 = (B / B_scale).round().clamp(-128, 127).to(torch.int8)
                
                _ = compiled_int8_matmul(A_int8, B_int8, A_scale, B_scale)
                results['INT8 (compiled)'] = benchmark(
                    lambda: compiled_int8_matmul(A_int8, B_int8, A_scale, B_scale)
                )
            except Exception as e:
                results['INT8 (compiled)'] = f"Error: {e}"
        
        # 5. Triton INT8
        if backends.get('triton') and TRITON_INT8_KERNEL is not None:
            try:
                A_scale = A.abs().max() / 127
                B_scale = B.abs().max() / 127
                A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
                B_int8 = (B / B_scale).round().clamp(-128, 127).to(torch.int8)
                
                results['INT8 (Triton)'] = benchmark(
                    lambda: triton_int8_matmul(A_int8, B_int8)
                )
            except Exception as e:
                results['INT8 (Triton)'] = f"Error: {e}"
        
        # 6. bitsandbytes
        if backends.get('bitsandbytes'):
            try:
                import bitsandbytes as bnb
                
                linear = bnb.nn.Linear8bitLt(K, N, bias=False, has_fp16_weights=False).cuda()
                linear.weight = bnb.nn.Int8Params(
                    B.t().contiguous(), requires_grad=False, has_fp16_weights=False
                ).cuda()
                
                results['INT8 (bitsandbytes)'] = benchmark(lambda: linear(A))
            except Exception as e:
                results['INT8 (bitsandbytes)'] = f"Error: {e}"
        
        # 7. FP8 (if supported)
        if backends.get('fp8') and check_fp8_support():
            try:
                results['FP8'] = benchmark(lambda: fp8_matmul(A, B))
            except Exception as e:
                results['FP8'] = f"Error: {e}"
        
        # Print results
        baseline = results['FP16']
        print(f"{'Method':<25} {'Time (ms)':<12} {'Speedup':<10}")
        for method, time_or_error in results.items():
            if isinstance(time_or_error, float):
                speedup = baseline / time_or_error
                print(f"{method:<25} {time_or_error:<12.4f} {speedup:<10.2f}x")
            else:
                print(f"{method:<25} {time_or_error}")
        
        # Cleanup
        del A, B
        torch.cuda.empty_cache()


def run_memory_benchmark():
    """Benchmark memory usage."""
    print("\n" + "=" * 70)
    print("Memory Usage Comparison")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    in_features = 4096
    out_features = 4096
    
    # FP16 weight
    weight_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)
    fp16_mem = weight_fp16.numel() * 2 / 1e6
    
    # INT8 weight
    weight_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8, device=device)
    int8_mem = weight_int8.numel() * 1 / 1e6
    
    # FP8 weight (if supported)
    if check_fp8_support():
        weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)
        fp8_mem = weight_fp8.numel() * 1 / 1e6
    else:
        fp8_mem = None
    
    print(f"\nWeight shape: ({out_features}, {in_features})")
    print(f"FP16:  {fp16_mem:.2f} MB (baseline)")
    print(f"INT8:  {int8_mem:.2f} MB ({fp16_mem/int8_mem:.1f}x compression)")
    if fp8_mem:
        print(f"FP8:   {fp8_mem:.2f} MB ({fp16_mem/fp8_mem:.1f}x compression)")


def main():
    """Main entry point."""
    backends = check_environment()
    
    # Setup Triton if available
    if backends.get('triton'):
        setup_triton_int8()
    
    run_comprehensive_benchmark(backends)
    run_memory_benchmark()
    
    print("\n" + "=" * 70)
    print("Recommendations for L20 (Ada Lovelace)")
    print("=" * 70)
    print("""
Based on your GPU (L20 / Ada Lovelace):

1. FP16 is very fast on L20's Tensor Cores
   - Often faster than INT8 due to overhead

2. FP8 is native to Ada Lovelace (if PyTorch supports it)
   - Better precision than INT8
   - Native hardware support
   - Try: torch.float8_e4m3fn

3. torch.compile can help
   - Use mode="reduce-overhead" for inference
   - Reduces kernel launch overhead

4. For Value reconstruction (small matrices):
   - FP16 is likely the best choice
   - INT8 overhead > compute savings

5. For large matrices (>=4096):
   - INT8/FP8 might be faster
   - Worth benchmarking

Best approach for your use case:
- Use FP16 for Value reconstruction
- Focus on other optimizations (FlashAttention, etc.)
""")


if __name__ == "__main__":
    main()
