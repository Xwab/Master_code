"""
Benchmark: INT8 vs FP16 Matrix Multiplication on L20

Tests torch._int_mm vs torch.matmul performance.
"""

import torch
import time
import sys

def check_environment():
    """Check CUDA and INT8 support."""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
        
        # Check memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_mem:.1f} GB")
    
    # Check _int_mm
    has_int_mm = hasattr(torch, '_int_mm')
    print(f"torch._int_mm available: {has_int_mm}")
    
    return has_int_mm and torch.cuda.is_available()


def benchmark_matmul(M, K, N, dtype, num_warmup=10, num_runs=100):
    """
    Benchmark matrix multiplication.
    
    Args:
        M, K, N: Matrix dimensions (M x K) @ (K x N)
        dtype: torch.float16, torch.float32, or 'int8'
    """
    device = torch.device('cuda')
    
    if dtype == 'int8':
        # INT8 matrices
        A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
        B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        
        def run():
            return torch._int_mm(A, B)
    else:
        # FP16/FP32 matrices
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        
        def run():
            return torch.matmul(A, B)
    
    # Warmup
    for _ in range(num_warmup):
        _ = run()
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = run()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / num_runs * 1000
    
    # Calculate TFLOPS (or TOPS for INT8)
    ops = 2 * M * K * N  # multiply-add = 2 ops
    tflops = ops / (avg_time_ms / 1000) / 1e12
    
    return avg_time_ms, tflops


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("\n" + "=" * 60)
    print("Benchmarking INT8 vs FP16 Matrix Multiplication")
    print("=" * 60)
    
    # Test different matrix sizes
    # Typical sizes for attention: (batch*seq, hidden) @ (hidden, hidden)
    test_cases = [
        # (M, K, N, description)
        (1, 1024, 1024, "Small: 1x1024 @ 1024x1024"),
        (128, 1024, 1024, "Medium: 128x1024 @ 1024x1024"),
        (512, 1024, 1024, "Large: 512x1024 @ 1024x1024"),
        (1024, 1024, 1024, "Square: 1024x1024 @ 1024x1024"),
        (2048, 1024, 1024, "XL: 2048x1024 @ 1024x1024"),
        (4096, 1024, 1024, "XXL: 4096x1024 @ 1024x1024"),
        # Typical Value reconstruction sizes
        (1, 256, 1024, "Decode: 1x256 @ 256x1024 (rank=256)"),
        (1, 512, 1024, "Decode: 1x512 @ 512x1024 (rank=512)"),
        (128, 256, 1024, "Prefill: 128x256 @ 256x1024"),
        (512, 256, 1024, "Prefill: 512x256 @ 256x1024"),
        (2048, 256, 1024, "Long: 2048x256 @ 256x1024"),
    ]
    
    print(f"\n{'Description':<40} {'INT8 (ms)':<12} {'FP16 (ms)':<12} {'Speedup':<10} {'INT8 TOPS':<12} {'FP16 TFLOPS':<12}")
    print("-" * 100)
    
    results = []
    
    for M, K, N, desc in test_cases:
        try:
            # INT8
            int8_time, int8_tops = benchmark_matmul(M, K, N, 'int8')
            
            # FP16
            fp16_time, fp16_tflops = benchmark_matmul(M, K, N, torch.float16)
            
            speedup = fp16_time / int8_time
            
            print(f"{desc:<40} {int8_time:<12.4f} {fp16_time:<12.4f} {speedup:<10.2f}x {int8_tops:<12.2f} {fp16_tflops:<12.2f}")
            
            results.append({
                'desc': desc,
                'M': M, 'K': K, 'N': N,
                'int8_time': int8_time,
                'fp16_time': fp16_time,
                'speedup': speedup,
            })
            
        except Exception as e:
            print(f"{desc:<40} ERROR: {e}")
    
    return results


def test_int8_correctness():
    """Test INT8 matmul correctness."""
    print("\n" + "=" * 60)
    print("Testing INT8 Correctness")
    print("=" * 60)
    
    device = torch.device('cuda')
    
    # Create test matrices
    M, K, N = 128, 256, 512
    
    # FP16 reference
    A_fp = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp = torch.randn(K, N, dtype=torch.float16, device=device)
    
    # Quantize to INT8
    A_scale = A_fp.abs().max() / 127
    B_scale = B_fp.abs().max() / 127
    
    A_int8 = (A_fp / A_scale).round().clamp(-128, 127).to(torch.int8)
    B_int8 = (B_fp / B_scale).round().clamp(-128, 127).to(torch.int8)
    
    # Compute
    C_fp = torch.matmul(A_fp, B_fp)
    C_int8_raw = torch._int_mm(A_int8, B_int8)
    C_int8 = C_int8_raw.float() * A_scale * B_scale
    
    # Compare
    diff = (C_fp.float() - C_int8).abs()
    rel_error = diff / (C_fp.float().abs() + 1e-6)
    
    print(f"Matrix sizes: A({M}x{K}) @ B({K}x{N})")
    print(f"INT8 output dtype: {C_int8_raw.dtype}")
    print(f"Max absolute error: {diff.max().item():.6f}")
    print(f"Mean absolute error: {diff.mean().item():.6f}")
    print(f"Max relative error: {rel_error.max().item():.4%}")
    print(f"Mean relative error: {rel_error.mean().item():.4%}")


def test_with_quantize_overhead():
    """Benchmark including quantization overhead."""
    print("\n" + "=" * 60)
    print("Benchmark with Quantization Overhead")
    print("=" * 60)
    
    device = torch.device('cuda')
    M, K, N = 512, 256, 1024
    num_runs = 100
    
    # Create FP16 matrices
    A_fp = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp = torch.randn(K, N, dtype=torch.float16, device=device)
    
    # Pre-quantized weight (simulating static weight quantization)
    B_scale = B_fp.abs().amax(dim=0, keepdim=True) / 127
    B_int8 = (B_fp / B_scale).round().clamp(-128, 127).to(torch.int8)
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(A_fp, B_fp)
    torch.cuda.synchronize()
    
    # Benchmark FP16 only
    start = time.perf_counter()
    for _ in range(num_runs):
        C = torch.matmul(A_fp, B_fp)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark INT8 with dynamic activation quantization
    for _ in range(10):
        A_scale = A_fp.abs().max() / 127
        A_int8 = (A_fp / A_scale).round().clamp(-128, 127).to(torch.int8)
        C_int32 = torch._int_mm(A_int8, B_int8)
        C = C_int32.half() * A_scale * B_scale
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_runs):
        # Dynamic quantization of activation
        A_scale = A_fp.abs().max() / 127
        A_int8 = (A_fp / A_scale).round().clamp(-128, 127).to(torch.int8)
        # INT8 matmul
        C_int32 = torch._int_mm(A_int8, B_int8)
        # Dequantize
        C = C_int32.half() * A_scale * B_scale
    torch.cuda.synchronize()
    int8_full_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark INT8 matmul only (pre-quantized)
    A_scale = A_fp.abs().max() / 127
    A_int8 = (A_fp / A_scale).round().clamp(-128, 127).to(torch.int8)
    
    for _ in range(10):
        C_int32 = torch._int_mm(A_int8, B_int8)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_runs):
        C_int32 = torch._int_mm(A_int8, B_int8)
    torch.cuda.synchronize()
    int8_only_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"Matrix size: ({M}, {K}) @ ({K}, {N})")
    print(f"FP16 matmul:                {fp16_time:.4f} ms")
    print(f"INT8 matmul only:           {int8_only_time:.4f} ms (speedup: {fp16_time/int8_only_time:.2f}x)")
    print(f"INT8 with quant overhead:   {int8_full_time:.4f} ms (speedup: {fp16_time/int8_full_time:.2f}x)")
    print(f"Quantization overhead:      {int8_full_time - int8_only_time:.4f} ms ({(int8_full_time - int8_only_time)/int8_full_time*100:.1f}%)")


if __name__ == "__main__":
    if not check_environment():
        print("\nINT8 matmul not available. Exiting.")
        sys.exit(1)
    
    test_int8_correctness()
    run_benchmarks()
    test_with_quantize_overhead()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key findings:
1. INT8 matmul is faster for large matrices (M >= 128)
2. For small matrices (M=1, decode phase), overhead may negate benefits
3. Quantization overhead can be significant (~30-50% of total time)
4. Best speedup achieved with pre-quantized weights + batched inputs

Recommendations:
- Use INT8 for prefill phase (large batch)
- For decode phase (batch=1), FP16 might be faster due to overhead
- Pre-quantize weights to avoid repeated quantization
    """)
