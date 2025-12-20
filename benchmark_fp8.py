"""
FP8 Benchmark for Ada Lovelace GPUs (L20, RTX 4090, etc.)

FP8 is native to Ada Lovelace and may provide better speedup than INT8.
"""

import torch
import torch.nn as nn
import time
import warnings

def check_fp8_availability():
    """Check FP8 support."""
    print("=" * 60)
    print("FP8 Availability Check")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check compute capability
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    
    if cap < (8, 9):
        print("✗ FP8 requires compute capability 8.9+ (Ada Lovelace)")
        return False
    
    print("✓ GPU supports FP8")
    
    # Check FP8 dtypes
    has_e4m3 = hasattr(torch, 'float8_e4m3fn')
    has_e5m2 = hasattr(torch, 'float8_e5m2')
    has_scaled_mm = hasattr(torch, '_scaled_mm')
    
    print(f"torch.float8_e4m3fn: {'✓' if has_e4m3 else '✗'}")
    print(f"torch.float8_e5m2: {'✓' if has_e5m2 else '✗'}")
    print(f"torch._scaled_mm: {'✓' if has_scaled_mm else '✗'}")
    
    return has_e4m3 and has_scaled_mm


def benchmark_fp8():
    """Benchmark FP8 vs FP16."""
    print("\n" + "=" * 60)
    print("FP8 vs FP16 Benchmark")
    print("=" * 60)
    
    device = torch.device('cuda')
    
    configs = [
        (128, 256, 1024, "V recon small"),
        (512, 256, 1024, "V recon med"),
        (2048, 256, 1024, "V recon long"),
        (512, 1024, 1024, "Medium"),
        (2048, 1024, 1024, "Large"),
        (4096, 4096, 4096, "XL"),
    ]
    
    num_warmup = 10
    num_runs = 100
    
    print(f"\n{'Config':<25} {'FP16 (ms)':<12} {'FP8 (ms)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for M, K, N, desc in configs:
        # Create FP16 data
        A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
        B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
        
        # FP16 benchmark
        def fp16_matmul():
            return torch.matmul(A_fp16, B_fp16)
        
        for _ in range(num_warmup):
            fp16_matmul()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_runs):
            fp16_matmul()
        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / num_runs * 1000
        
        # Try FP8
        try:
            # Quantize to FP8
            A_scale = A_fp16.abs().amax() / 448.0 + 1e-12
            B_scale = B_fp16.abs().amax() / 448.0 + 1e-12
            
            A_fp8 = (A_fp16 / A_scale).to(torch.float8_e4m3fn)
            B_fp8 = (B_fp16 / B_scale).to(torch.float8_e4m3fn)
            
            # Scale tensors need to be on GPU as scalar tensors
            A_scale_t = torch.tensor(A_scale.item(), dtype=torch.float32, device=device)
            B_scale_t = torch.tensor(B_scale.item(), dtype=torch.float32, device=device)
            
            def fp8_matmul():
                return torch._scaled_mm(
                    A_fp8, B_fp8,
                    scale_a=A_scale_t,
                    scale_b=B_scale_t,
                    out_dtype=torch.float16,
                )
            
            # Warmup
            for _ in range(num_warmup):
                fp8_matmul()
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(num_runs):
                fp8_matmul()
            torch.cuda.synchronize()
            fp8_time = (time.perf_counter() - start) / num_runs * 1000
            
            speedup = fp16_time / fp8_time
            print(f"{desc:<25} {fp16_time:<12.4f} {fp8_time:<12.4f} {speedup:<10.2f}x")
            
        except Exception as e:
            print(f"{desc:<25} {fp16_time:<12.4f} {'Error':<12} {str(e)[:30]}")
        
        del A_fp16, B_fp16
        torch.cuda.empty_cache()


def benchmark_transformer_float8():
    """Test torch.ao.float8 if available (experimental)."""
    print("\n" + "=" * 60)
    print("TransformerEngine-style Float8 (if available)")
    print("=" * 60)
    
    try:
        # Check for transformer_engine (NVIDIA's library)
        import transformer_engine.pytorch as te
        print("✓ TransformerEngine available")
        
        device = torch.device('cuda')
        
        # Create a linear layer
        linear_fp16 = nn.Linear(1024, 1024, bias=False).half().cuda()
        linear_fp8 = te.Linear(1024, 1024, bias=False).cuda()
        linear_fp8.weight.data = linear_fp16.weight.data.clone()
        
        x = torch.randn(512, 1024, dtype=torch.float16, device=device)
        
        # Warmup
        for _ in range(10):
            _ = linear_fp16(x)
            with te.fp8_autocast():
                _ = linear_fp8(x)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = linear_fp16(x)
        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / 100 * 1000
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            with te.fp8_autocast():
                _ = linear_fp8(x)
        torch.cuda.synchronize()
        fp8_time = (time.perf_counter() - start) / 100 * 1000
        
        print(f"FP16: {fp16_time:.4f} ms")
        print(f"FP8 (TE): {fp8_time:.4f} ms")
        print(f"Speedup: {fp16_time/fp8_time:.2f}x")
        
    except ImportError:
        print("TransformerEngine not installed")
        print("Install: pip install transformer_engine")


def main():
    """Main entry point."""
    warnings.filterwarnings('ignore')
    
    if not check_fp8_availability():
        print("\nFP8 not fully available. Possible solutions:")
        print("1. Upgrade PyTorch: pip install torch>=2.1")
        print("2. Install TransformerEngine: pip install transformer_engine")
        return
    
    benchmark_fp8()
    benchmark_transformer_float8()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
FP8 on Ada Lovelace:
- E4M3: Higher precision, good for forward pass
- E5M2: Lower precision, good for gradients

For Value reconstruction:
- FP8 might provide speedup for large matrices
- For small matrices, FP16 is often faster

Recommendations:
1. If FP8 shows speedup > 1.2x, consider using it
2. Otherwise, stick with FP16
3. Consider TransformerEngine for production
""")


if __name__ == "__main__":
    main()
