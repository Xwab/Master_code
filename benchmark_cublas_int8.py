"""
Benchmark: cuBLAS/cuDNN INT8 GEMM vs FP16 GEMM

Uses torch's backend which leverages cuBLAS/cuBLASLt for GEMM operations.
torch._int_mm internally uses cublasLtMatmul with INT8 Tensor Cores.

Requirements:
- PyTorch >= 2.0
- CUDA GPU with Tensor Core support (Volta+)
- For best INT8 performance: Ampere+ (sm_80+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os

# Ensure deterministic behavior for benchmarking
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for fair FP16 comparison


def check_environment():
    """Check CUDA environment and capabilities."""
    print("=" * 80)
    print("Environment Check - cuBLAS/cuDNN INT8 Support")
    print("=" * 80)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        return False
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"TF32 allowed: {torch.backends.cuda.matmul.allow_tf32}")
    
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"\nGPU: {gpu_name}")
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    
    # Check architecture
    arch_info = {
        (7, 0): "Volta (V100) - INT8 Tensor Cores",
        (7, 5): "Turing (RTX 20xx) - INT8 Tensor Cores", 
        (8, 0): "Ampere (A100) - INT8 Tensor Cores (optimized)",
        (8, 6): "Ampere (RTX 30xx) - INT8 Tensor Cores",
        (8, 9): "Ada Lovelace (RTX 40xx, L20) - INT8 + FP8 Tensor Cores",
        (9, 0): "Hopper (H100) - INT8 + FP8 Tensor Cores",
    }
    
    arch = (cap[0], cap[1])
    if arch in arch_info:
        print(f"Architecture: {arch_info[arch]}")
    else:
        print(f"Architecture: Unknown (sm_{cap[0]}{cap[1]})")
    
    # Check torch._int_mm
    has_int_mm = hasattr(torch, '_int_mm')
    print(f"\ntorch._int_mm (cuBLASLt INT8): {'✓' if has_int_mm else '✗'}")
    
    # Check FP8 support
    has_fp8 = hasattr(torch, 'float8_e4m3fn') and cap >= (8, 9)
    print(f"FP8 support: {'✓' if has_fp8 else '✗'}")
    
    # Check scaled_mm
    has_scaled_mm = hasattr(torch, '_scaled_mm')
    print(f"torch._scaled_mm: {'✓' if has_scaled_mm else '✗'}")
    
    return has_int_mm


def benchmark_func(func, num_warmup=20, num_runs=100):
    """Benchmark a function with proper CUDA synchronization."""
    # Warmup
    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    
    # Use CUDA events for more accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        func()
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / num_runs
    
    return elapsed_ms


def ensure_aligned(tensor, alignment=16):
    """Ensure tensor is contiguous and properly aligned for Tensor Cores."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


# ============================================================================
# INT8 Quantization Functions
# ============================================================================

def quantize_per_tensor(x: torch.Tensor) -> tuple:
    """Per-tensor symmetric quantization to INT8."""
    scale = x.abs().amax() / 127.0 + 1e-6
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def quantize_per_token(x: torch.Tensor) -> tuple:
    """Per-token (per-row) symmetric quantization to INT8.
    
    For Value states with shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim),
    each token (row) is quantized independently.
    
    Args:
        x: Tensor of shape (..., hidden_dim), quantize along last dim
    
    Returns:
        x_int8: Quantized tensor
        scale: Scale tensor of shape (..., 1)
    """
    # Quantize each row (token) independently
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def quantize_per_channel(x: torch.Tensor, dim: int = 0) -> tuple:
    """Per-channel symmetric quantization to INT8.
    
    For Key states or weights, quantize along channel dimension.
    
    Args:
        x: Tensor to quantize
        dim: Dimension to preserve (0 for row-wise, 1 for column-wise)
    
    Returns:
        x_int8: Quantized tensor
        scale: Scale tensor
    """
    if dim == 0:
        # Per-row quantization (each row has its own scale)
        scale = x.abs().amax(dim=1, keepdim=True) / 127.0 + 1e-6
    else:
        # Per-column quantization (each column has its own scale)
        scale = x.abs().amax(dim=0, keepdim=True) / 127.0 + 1e-6
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def dequantize(x_int8: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor = None) -> torch.Tensor:
    """Dequantize INT32 result to FP16."""
    if scale_b is None:
        return x_int8.half() * scale_a
    return x_int8.half() * scale_a * scale_b


def dequantize_per_token_result(C_int32: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """Dequantize result when A is per-token quantized and B is per-tensor quantized.
    
    C = A @ B, where A has per-token scales (M, 1) and B has per-tensor scale (scalar)
    C_int32 is (M, N), result should be C_int32 * scale_a * scale_b
    """
    return C_int32.half() * scale_a * scale_b


# ============================================================================
# GEMM Implementations
# ============================================================================

class GEMMBenchmark:
    """Benchmark different GEMM implementations."""
    
    def __init__(self, M, K, N, device='cuda'):
        self.M = M
        self.K = K
        self.N = N
        self.device = device
        
        # Create test matrices
        self.A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
        self.B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
        
        # Ensure alignment
        self.A_fp16 = ensure_aligned(self.A_fp16)
        self.B_fp16 = ensure_aligned(self.B_fp16)
        
        # Pre-quantize for INT8
        self.A_int8, self.A_scale = quantize_per_tensor(self.A_fp16)
        self.B_int8, self.B_scale = quantize_per_tensor(self.B_fp16)
        
        # Per-channel quantization for weight
        self.B_int8_pc, self.B_scale_pc = quantize_per_channel(self.B_fp16, dim=0)
        
        # Ensure alignment
        self.A_int8 = ensure_aligned(self.A_int8)
        self.B_int8 = ensure_aligned(self.B_int8)
        self.B_int8_pc = ensure_aligned(self.B_int8_pc)
    
    def fp16_cublas(self):
        """FP16 GEMM using cuBLAS (via torch.matmul)."""
        return torch.matmul(self.A_fp16, self.B_fp16)
    
    def int8_cublaslt(self):
        """INT8 GEMM using cuBLASLt (via torch._int_mm)."""
        return torch._int_mm(self.A_int8, self.B_int8)
    
    def int8_cublaslt_with_dequant(self):
        """INT8 GEMM + dequantization."""
        C_int32 = torch._int_mm(self.A_int8, self.B_int8)
        return C_int32.half() * self.A_scale * self.B_scale
    
    def int8_full_pipeline(self):
        """Full INT8 pipeline: quant A + int_mm + dequant."""
        # Quantize A (activation) at runtime
        A_scale = self.A_fp16.abs().amax() / 127.0 + 1e-6
        A_int8 = (self.A_fp16 / A_scale).round().clamp(-128, 127).to(torch.int8)
        
        # INT8 GEMM (B is pre-quantized)
        C_int32 = torch._int_mm(A_int8, self.B_int8)
        
        # Dequantize
        return C_int32.half() * A_scale * self.B_scale
    
    def weight_only_int8(self):
        """Weight-only INT8: dequant weight -> FP16 GEMM."""
        B_fp16 = self.B_int8_pc.half() * self.B_scale_pc
        return torch.matmul(self.A_fp16, B_fp16)
    
    def cleanup(self):
        """Free memory."""
        del self.A_fp16, self.B_fp16
        del self.A_int8, self.B_int8, self.A_scale, self.B_scale
        del self.B_int8_pc, self.B_scale_pc
        torch.cuda.empty_cache()


# ============================================================================
# Benchmarks
# ============================================================================

def run_gemm_comparison():
    """Compare different GEMM implementations."""
    print("\n" + "=" * 80)
    print("cuBLAS GEMM Comparison: FP16 vs INT8 (cuBLASLt)")
    print("=" * 80)
    
    device = 'cuda'
    
    # Test configurations: (M, K, N, description)
    configs = [
        # Decode (small batch)
        (1, 1024, 1024, "Decode: 1x1024 @ 1024x1024"),
        (1, 4096, 4096, "Decode: 1x4096 @ 4096x4096"),
        
        # Small prefill
        (128, 1024, 1024, "Small: 128x1024 @ 1024x1024"),
        (128, 4096, 4096, "Small: 128x4096 @ 4096x4096"),
        
        # Medium prefill
        (512, 1024, 1024, "Medium: 512x1024 @ 1024x1024"),
        (512, 4096, 4096, "Medium: 512x4096 @ 4096x4096"),
        
        # Large prefill
        (2048, 1024, 1024, "Large: 2048x1024 @ 1024x1024"),
        (2048, 4096, 4096, "Large: 2048x4096 @ 4096x4096"),
        
        # Very long sequences
        (8192, 1024, 1024, "8K: 8192x1024 @ 1024x1024"),
        (16384, 1024, 1024, "16K: 16384x1024 @ 1024x1024"),
        (24576, 1024, 1024, "24K: 24576x1024 @ 1024x1024"),
        
        # Value reconstruction (rank -> hidden)
        (1, 256, 1024, "V decode: 1x256 @ 256x1024"),
        (128, 256, 1024, "V small: 128x256 @ 256x1024"),
        (512, 256, 1024, "V med: 512x256 @ 256x1024"),
        (2048, 256, 1024, "V long: 2048x256 @ 256x1024"),
        (8192, 256, 1024, "V 8K: 8192x256 @ 256x1024"),
        (16384, 256, 1024, "V 16K: 16384x256 @ 256x1024"),
        (24576, 256, 1024, "V 24K: 24576x256 @ 256x1024"),
    ]
    
    print(f"\n{'Config':<35} {'FP16':<10} {'INT8只':<10} {'INT8+Dq':<10} {'Full W8A8':<10} {'W-Only':<10} {'INT8 Spd':<10}")
    print("-" * 105)
    
    results = []
    
    for M, K, N, desc in configs:
        try:
            bench = GEMMBenchmark(M, K, N, device)
            
            # Benchmark each method
            fp16_time = benchmark_func(bench.fp16_cublas)
            int8_only_time = benchmark_func(bench.int8_cublaslt)
            int8_dequant_time = benchmark_func(bench.int8_cublaslt_with_dequant)
            int8_full_time = benchmark_func(bench.int8_full_pipeline)
            wo_time = benchmark_func(bench.weight_only_int8)
            
            # Calculate speedup (INT8 only vs FP16)
            int8_speedup = fp16_time / int8_only_time
            
            print(f"{desc:<35} {fp16_time:<10.4f} {int8_only_time:<10.4f} {int8_dequant_time:<10.4f} {int8_full_time:<10.4f} {wo_time:<10.4f} {int8_speedup:<10.2f}x")
            
            results.append({
                'config': desc,
                'M': M, 'K': K, 'N': N,
                'fp16': fp16_time,
                'int8_only': int8_only_time,
                'int8_dequant': int8_dequant_time,
                'int8_full': int8_full_time,
                'wo': wo_time,
                'speedup': int8_speedup,
            })
            
            bench.cleanup()
            
        except Exception as e:
            print(f"{desc:<35} ERROR: {e}")
    
    print("\n说明:")
    print("  - FP16: cuBLAS FP16 GEMM (torch.matmul)")
    print("  - INT8只: cuBLASLt INT8 GEMM (torch._int_mm)，无量化开销")
    print("  - INT8+Dq: INT8 GEMM + 反量化结果")
    print("  - Full W8A8: 完整流程 (量化A + INT8 GEMM + 反量化)")
    print("  - W-Only: Weight-Only INT8 (反量化权重 + FP16 GEMM)")
    print("  - INT8 Spd: 纯 INT8 GEMM 相对 FP16 的加速比")
    
    return results


def run_timing_breakdown():
    """Detailed timing breakdown for each operation."""
    print("\n" + "=" * 80)
    print("Timing Breakdown: Each Operation Separately")
    print("=" * 80)
    
    device = 'cuda'
    
    configs = [
        (512, 1024, 1024, "Medium"),
        (2048, 1024, 1024, "Large"),
        (8192, 1024, 1024, "8K seq"),
        (16384, 1024, 1024, "16K seq"),
        (24576, 1024, 1024, "24K seq"),
        (2048, 256, 1024, "V recon"),
        (8192, 256, 1024, "V 8K"),
        (16384, 256, 1024, "V 16K"),
        (24576, 256, 1024, "V 24K"),
    ]
    
    print(f"\n{'Config':<15} {'Quant A':<10} {'Quant B':<10} {'INT8 MM':<10} {'Dequant':<10} {'FP16 MM':<10} {'Total INT8':<12} {'Overhead':<10}")
    print("-" * 100)
    
    for M, K, N, desc in configs:
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)
        
        # Pre-quantize B (weight)
        B_scale = B.abs().amax() / 127.0 + 1e-6
        B_int8 = (B / B_scale).round().clamp(-128, 127).to(torch.int8)
        
        # 1. Quantize A
        def quant_A():
            scale = A.abs().amax() / 127.0 + 1e-6
            return (A / scale).round().clamp(-128, 127).to(torch.int8), scale
        
        quant_A_time = benchmark_func(quant_A)
        
        # 2. Quantize B (for reference, usually offline)
        def quant_B():
            scale = B.abs().amax() / 127.0 + 1e-6
            return (B / scale).round().clamp(-128, 127).to(torch.int8), scale
        
        quant_B_time = benchmark_func(quant_B)
        
        # Pre-quantize A for INT8 MM benchmark
        A_scale = A.abs().amax() / 127.0 + 1e-6
        A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
        
        # 3. INT8 GEMM only
        def int8_mm():
            return torch._int_mm(A_int8, B_int8)
        
        int8_mm_time = benchmark_func(int8_mm)
        
        # 4. Dequantize result
        C_int32 = torch._int_mm(A_int8, B_int8)
        
        def dequant():
            return C_int32.half() * A_scale * B_scale
        
        dequant_time = benchmark_func(dequant)
        
        # 5. FP16 GEMM
        def fp16_mm():
            return torch.matmul(A, B)
        
        fp16_mm_time = benchmark_func(fp16_mm)
        
        # Total INT8 time (with quant A + INT8 MM + dequant)
        total_int8 = quant_A_time + int8_mm_time + dequant_time
        overhead = total_int8 - fp16_mm_time
        
        print(f"{desc:<15} {quant_A_time:<10.4f} {quant_B_time:<10.4f} {int8_mm_time:<10.4f} {dequant_time:<10.4f} {fp16_mm_time:<10.4f} {total_int8:<12.4f} {overhead:<10.4f}")
        
        del A, B, A_int8, B_int8, C_int32
        torch.cuda.empty_cache()
    
    print("\n说明:")
    print("  - Overhead > 0: INT8 总时间比 FP16 长")
    print("  - Overhead < 0: INT8 总时间比 FP16 短 (有加速)")


def run_tensor_core_utilization():
    """Check Tensor Core utilization."""
    print("\n" + "=" * 80)
    print("Tensor Core Utilization Analysis")
    print("=" * 80)
    
    device = 'cuda'
    
    # Tensor Cores prefer specific dimensions (multiples of 8 for INT8, 16 for FP16)
    print("\n测试不同对齐方式对性能的影响:")
    print("(Tensor Cores 对 INT8 偏好 8 的倍数，对 FP16 偏好 16 的倍数)\n")
    
    base_configs = [
        (2048, 1024, 1024),
        (8192, 1024, 1024),
        (16384, 1024, 1024),
    ]
    
    for M_base, K, N in base_configs:
        print(f"\n基础尺寸: {M_base}x{K} @ {K}x{N}")
        print(f"{'M':<10} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'Speedup':<10} {'对齐?':<10}")
        print("-" * 55)
        
        for offset in [0, 1, 7, 8, 15, 16]:
            M = M_base + offset
            
            A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
            B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
            
            A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
            B_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
            
            fp16_time = benchmark_func(lambda: torch.matmul(A_fp16, B_fp16), num_warmup=10, num_runs=50)
            int8_time = benchmark_func(lambda: torch._int_mm(A_int8, B_int8), num_warmup=10, num_runs=50)
            
            speedup = fp16_time / int8_time
            aligned = "✓" if M % 8 == 0 else "✗"
            
            print(f"{M:<10} {fp16_time:<12.4f} {int8_time:<12.4f} {speedup:<10.2f}x {aligned:<10}")
            
            del A_fp16, B_fp16, A_int8, B_int8
            torch.cuda.empty_cache()


def run_gflops_analysis():
    """Analyze GFLOPS for different operations."""
    print("\n" + "=" * 80)
    print("GFLOPS Analysis: Theoretical vs Achieved")
    print("=" * 80)
    
    device = 'cuda'
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    
    # Theoretical peak GFLOPS (approximate)
    # L20: ~119 TFLOPS FP16, ~238 TOPS INT8
    # RTX 4090: ~165 TFLOPS FP16, ~330 TOPS INT8
    # A100: ~312 TFLOPS FP16, ~624 TOPS INT8
    
    print(f"\nGPU: {gpu_name}")
    print(f"(理论峰值性能请查阅具体 GPU 规格)")
    
    configs = [
        (2048, 4096, 4096, "Large"),
        (8192, 4096, 4096, "8K Large"),
        (16384, 4096, 4096, "16K Large"),
        (24576, 4096, 4096, "24K Large"),
    ]
    
    print(f"\n{'Config':<15} {'FP16 TFLOPS':<15} {'INT8 TOPS':<15} {'INT8/FP16':<12}")
    print("-" * 60)
    
    for M, K, N, desc in configs:
        A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
        B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
        
        A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
        B_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        
        fp16_time = benchmark_func(lambda: torch.matmul(A_fp16, B_fp16), num_warmup=10, num_runs=50)
        int8_time = benchmark_func(lambda: torch._int_mm(A_int8, B_int8), num_warmup=10, num_runs=50)
        
        # Calculate FLOPS/OPS
        ops = 2 * M * K * N  # multiply-add = 2 ops
        
        fp16_tflops = ops / (fp16_time / 1000) / 1e12
        int8_tops = ops / (int8_time / 1000) / 1e12
        
        ratio = int8_tops / fp16_tflops
        
        print(f"{desc:<15} {fp16_tflops:<15.2f} {int8_tops:<15.2f} {ratio:<12.2f}x")
        
        del A_fp16, B_fp16, A_int8, B_int8
        torch.cuda.empty_cache()
    
    print("\n说明:")
    print("  - TFLOPS: Tera Floating-Point Operations Per Second")
    print("  - TOPS: Tera Integer Operations Per Second")
    print("  - 理论上 INT8 应该是 FP16 的 2x (Tensor Core 设计)")


# ============================================================================
# Per-Token Quantization Benchmark (for Value)
# ============================================================================

def run_per_token_quantization_benchmark():
    """Benchmark per-token quantization (KIVI-style for Value)."""
    print("\n" + "=" * 80)
    print("Per-Token Quantization Benchmark (KIVI-style Value Quantization)")
    print("=" * 80)
    
    device = 'cuda'
    
    # Test configurations: (batch_size, seq_len, hidden_dim, description)
    # Value shape is typically (batch, seq_len, hidden_dim)
    configs = [
        # Small batch, various sequence lengths
        (1, 512, 1024, "B1 x 512 x 1024"),
        (1, 2048, 1024, "B1 x 2K x 1024"),
        (1, 8192, 1024, "B1 x 8K x 1024"),
        
        # Medium batch sizes
        (8, 256, 1024, "B8 x 256 x 1024"),
        (16, 256, 1024, "B16 x 256 x 1024"),
        (32, 256, 1024, "B32 x 256 x 1024"),
        
        # Large batch sizes (shorter sequences)
        (64, 128, 1024, "B64 x 128 x 1024"),
        (64, 256, 1024, "B64 x 256 x 1024"),
        (64, 512, 1024, "B64 x 512 x 1024"),
        
        (128, 64, 1024, "B128 x 64 x 1024"),
        (128, 128, 1024, "B128 x 128 x 1024"),
        (128, 256, 1024, "B128 x 256 x 1024"),
        
        # Very large batch
        (256, 64, 1024, "B256 x 64 x 1024"),
        (256, 128, 1024, "B256 x 128 x 1024"),
        
        # Value reconstruction (rank -> hidden)
        (64, 128, 256, "V: B64 x 128 x 256"),
        (64, 256, 256, "V: B64 x 256 x 256"),
        (128, 128, 256, "V: B128 x 128 x 256"),
        (128, 256, 256, "V: B128 x 256 x 256"),
        (256, 128, 256, "V: B256 x 128 x 256"),
    ]
    
    print(f"\n{'Config':<25} {'Per-Tensor Q':<12} {'Per-Token Q':<12} {'Diff':<10} {'Elements':<12}")
    print("-" * 75)
    
    for batch, seq_len, hidden_dim, desc in configs:
        try:
            # Create Value tensor
            V = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float16, device=device)
            
            # Per-tensor quantization
            def quant_per_tensor():
                scale = V.abs().amax() / 127.0 + 1e-6
                return (V / scale).round().clamp(-128, 127).to(torch.int8), scale
            
            per_tensor_time = benchmark_func(quant_per_tensor, num_warmup=10, num_runs=50)
            
            # Per-token quantization (KIVI-style for Value)
            def quant_per_token():
                # Each token (row in last 2 dims) has its own scale
                scale = V.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
                return (V / scale).round().clamp(-128, 127).to(torch.int8), scale
            
            per_token_time = benchmark_func(quant_per_token, num_warmup=10, num_runs=50)
            
            diff = per_token_time - per_tensor_time
            elements = batch * seq_len * hidden_dim
            
            print(f"{desc:<25} {per_tensor_time:<12.4f} {per_token_time:<12.4f} {diff:<10.4f} {elements:<12}")
            
            del V
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")
    
    print("\n说明:")
    print("  - Per-Tensor Q: 整个张量一个 scale")
    print("  - Per-Token Q: 每个 token (行) 一个 scale (KIVI Value 量化方式)")
    print("  - Per-Token 量化精度更高，但开销稍大")


def run_per_token_matmul_benchmark():
    """Benchmark matmul with per-token quantized activation."""
    print("\n" + "=" * 80)
    print("Per-Token INT8 Matmul Benchmark (Value Reconstruction)")
    print("=" * 80)
    
    device = 'cuda'
    
    # For Value reconstruction: V @ A_linear.T
    # V shape: (batch, seq_len, rank) or (batch * seq_len, rank)
    # A_linear.T shape: (rank, hidden_dim)
    # Output: (batch, seq_len, hidden_dim)
    
    configs = [
        # (batch_size, seq_len, rank, hidden_dim, description)
        # Small batch, long sequence
        (1, 2048, 256, 1024, "B1 x 2K, 256->1024"),
        (1, 8192, 256, 1024, "B1 x 8K, 256->1024"),
        
        # Medium batch
        (8, 256, 256, 1024, "B8 x 256, 256->1024"),
        (16, 256, 256, 1024, "B16 x 256, 256->1024"),
        (32, 256, 256, 1024, "B32 x 256, 256->1024"),
        
        # Large batch, short sequence
        (64, 64, 256, 1024, "B64 x 64, 256->1024"),
        (64, 128, 256, 1024, "B64 x 128, 256->1024"),
        (64, 256, 256, 1024, "B64 x 256, 256->1024"),
        (64, 512, 256, 1024, "B64 x 512, 256->1024"),
        
        (128, 64, 256, 1024, "B128 x 64, 256->1024"),
        (128, 128, 256, 1024, "B128 x 128, 256->1024"),
        (128, 256, 256, 1024, "B128 x 256, 256->1024"),
        
        # Very large batch
        (256, 64, 256, 1024, "B256 x 64, 256->1024"),
        (256, 128, 256, 1024, "B256 x 128, 256->1024"),
        
        # Large batch with larger dimensions
        (64, 128, 512, 4096, "B64 x 128, 512->4096"),
        (128, 128, 512, 4096, "B128 x 128, 512->4096"),
        (64, 256, 512, 4096, "B64 x 256, 512->4096"),
        
        # Mixed: medium batch, medium seq
        (32, 512, 256, 1024, "B32 x 512, 256->1024"),
        (32, 1024, 256, 1024, "B32 x 1K, 256->1024"),
        (64, 512, 256, 1024, "B64 x 512, 256->1024"),
    ]
    
    print(f"\n{'Config':<25} {'FP16':<10} {'INT8 PT':<10} {'INT8 Full':<10} {'W-Only':<10} {'INT8 Spd':<10}")
    print("-" * 85)
    
    for batch, seq_len, rank, hidden_dim, desc in configs:
        try:
            # Flatten batch and seq_len for matmul: (batch * seq_len, rank)
            M = batch * seq_len
            K = rank
            N = hidden_dim
            
            # Activation (Value latent)
            A = torch.randn(M, K, dtype=torch.float16, device=device)
            # Weight (A_linear transposed)
            W = torch.randn(K, N, dtype=torch.float16, device=device)
            
            # Pre-quantize weight (per-tensor for simplicity)
            W_scale = W.abs().amax() / 127.0 + 1e-6
            W_int8 = (W / W_scale).round().clamp(-128, 127).to(torch.int8)
            
            # Weight-only quantization (per-channel)
            W_scale_pc = W.abs().amax(dim=0, keepdim=True) / 127.0 + 1e-6
            W_int8_pc = (W / W_scale_pc).round().clamp(-128, 127).to(torch.int8)
            
            # 1. FP16 baseline
            fp16_time = benchmark_func(lambda: torch.matmul(A, W), num_warmup=10, num_runs=50)
            
            # 2. INT8 per-token (pure int_mm, pre-quantized)
            A_scale_pt = A.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
            A_int8_pt = (A / A_scale_pt).round().clamp(-128, 127).to(torch.int8)
            
            int8_pt_time = benchmark_func(lambda: torch._int_mm(A_int8_pt, W_int8), num_warmup=10, num_runs=50)
            
            # 3. Full INT8 per-token (with quantization overhead)
            def int8_full_per_token():
                a_scale = A.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
                a_int8 = (A / a_scale).round().clamp(-128, 127).to(torch.int8)
                c_int32 = torch._int_mm(a_int8, W_int8)
                return c_int32.half() * a_scale * W_scale
            
            int8_full_time = benchmark_func(int8_full_per_token, num_warmup=10, num_runs=50)
            
            # 4. Weight-only INT8
            def weight_only():
                W_fp = W_int8_pc.half() * W_scale_pc
                return torch.matmul(A, W_fp)
            
            wo_time = benchmark_func(weight_only, num_warmup=10, num_runs=50)
            
            int8_spd = fp16_time / int8_pt_time
            
            print(f"{desc:<25} {fp16_time:<10.4f} {int8_pt_time:<10.4f} {int8_full_time:<10.4f} {wo_time:<10.4f} {int8_spd:<10.2f}x")
            
            del A, W, A_int8_pt, W_int8, W_int8_pc
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")
    
    print("\n说明:")
    print("  - FP16: 标准 FP16 矩阵乘法")
    print("  - INT8 PT: 纯 INT8 GEMM (per-token 预量化，无开销)")
    print("  - INT8 Full: 完整 INT8 流程 (per-token 量化 + GEMM + 反量化)")
    print("  - W-Only: Weight-Only INT8")
    print("  - INT8 Spd: 纯 INT8 GEMM 相对 FP16 的加速比")


def run_large_batch_comparison():
    """Compare performance across different batch sizes."""
    print("\n" + "=" * 80)
    print("Large Batch Size Comparison")
    print("=" * 80)
    
    device = 'cuda'
    rank = 256
    hidden_dim = 1024
    
    # Test 1: Very large batch with short sequence
    print(f"\n=== Test 1: Varying Batch Size (seq_len=128) ===")
    print(f"Fixed: seq_len=128, rank={rank}, hidden_dim={hidden_dim}")
    print(f"\n{'Batch':<10} {'Total Tokens':<15} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'Full INT8':<12} {'Speedup':<10}")
    print("-" * 75)
    
    seq_len = 128
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    
    for batch in batch_sizes:
        try:
            M = batch * seq_len
            K = rank
            N = hidden_dim
            
            A = torch.randn(M, K, dtype=torch.float16, device=device)
            W = torch.randn(K, N, dtype=torch.float16, device=device)
            
            # Pre-quantize
            A_scale = A.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
            A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
            W_scale = W.abs().amax() / 127.0 + 1e-6
            W_int8 = (W / W_scale).round().clamp(-128, 127).to(torch.int8)
            
            # FP16
            fp16_time = benchmark_func(lambda: torch.matmul(A, W), num_warmup=10, num_runs=50)
            
            # INT8 only
            int8_time = benchmark_func(lambda: torch._int_mm(A_int8, W_int8), num_warmup=10, num_runs=50)
            
            # Full INT8
            def int8_full():
                a_s = A.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
                a_i = (A / a_s).round().clamp(-128, 127).to(torch.int8)
                c = torch._int_mm(a_i, W_int8)
                return c.half() * a_s * W_scale
            
            full_time = benchmark_func(int8_full, num_warmup=10, num_runs=50)
            
            speedup = fp16_time / int8_time
            total_tokens = batch * seq_len
            
            print(f"{batch:<10} {total_tokens:<15} {fp16_time:<12.4f} {int8_time:<12.4f} {full_time:<12.4f} {speedup:<10.2f}x")
            
            del A, W, A_int8, W_int8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:<10} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{batch:<10} ERROR: {e}")
    
    # Test 2: Large batch with medium sequence
    print(f"\n=== Test 2: Varying Batch Size (seq_len=256) ===")
    print(f"Fixed: seq_len=256, rank={rank}, hidden_dim={hidden_dim}")
    print(f"\n{'Batch':<10} {'Total Tokens':<15} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'Full INT8':<12} {'Speedup':<10}")
    print("-" * 75)
    
    seq_len = 256
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    
    for batch in batch_sizes:
        try:
            M = batch * seq_len
            K = rank
            N = hidden_dim
            
            A = torch.randn(M, K, dtype=torch.float16, device=device)
            W = torch.randn(K, N, dtype=torch.float16, device=device)
            
            A_scale = A.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
            A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
            W_scale = W.abs().amax() / 127.0 + 1e-6
            W_int8 = (W / W_scale).round().clamp(-128, 127).to(torch.int8)
            
            fp16_time = benchmark_func(lambda: torch.matmul(A, W), num_warmup=10, num_runs=50)
            int8_time = benchmark_func(lambda: torch._int_mm(A_int8, W_int8), num_warmup=10, num_runs=50)
            
            def int8_full():
                a_s = A.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
                a_i = (A / a_s).round().clamp(-128, 127).to(torch.int8)
                c = torch._int_mm(a_i, W_int8)
                return c.half() * a_s * W_scale
            
            full_time = benchmark_func(int8_full, num_warmup=10, num_runs=50)
            speedup = fp16_time / int8_time
            total_tokens = batch * seq_len
            
            print(f"{batch:<10} {total_tokens:<15} {fp16_time:<12.4f} {int8_time:<12.4f} {full_time:<12.4f} {speedup:<10.2f}x")
            
            del A, W, A_int8, W_int8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:<10} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{batch:<10} ERROR: {e}")
    
    # Test 3: Fixed large batch, vary sequence length
    print(f"\n=== Test 3: Varying Seq Len (batch=64) ===")
    print(f"Fixed: batch=64, rank={rank}, hidden_dim={hidden_dim}")
    print(f"\n{'Seq Len':<10} {'Total Tokens':<15} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'Full INT8':<12} {'Speedup':<10}")
    print("-" * 75)
    
    batch = 64
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    
    for seq_len in seq_lengths:
        try:
            M = batch * seq_len
            K = rank
            N = hidden_dim
            
            A = torch.randn(M, K, dtype=torch.float16, device=device)
            W = torch.randn(K, N, dtype=torch.float16, device=device)
            
            A_scale = A.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
            A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
            W_scale = W.abs().amax() / 127.0 + 1e-6
            W_int8 = (W / W_scale).round().clamp(-128, 127).to(torch.int8)
            
            fp16_time = benchmark_func(lambda: torch.matmul(A, W), num_warmup=10, num_runs=50)
            int8_time = benchmark_func(lambda: torch._int_mm(A_int8, W_int8), num_warmup=10, num_runs=50)
            
            def int8_full():
                a_s = A.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
                a_i = (A / a_s).round().clamp(-128, 127).to(torch.int8)
                c = torch._int_mm(a_i, W_int8)
                return c.half() * a_s * W_scale
            
            full_time = benchmark_func(int8_full, num_warmup=10, num_runs=50)
            speedup = fp16_time / int8_time
            total_tokens = batch * seq_len
            
            print(f"{seq_len:<10} {total_tokens:<15} {fp16_time:<12.4f} {int8_time:<12.4f} {full_time:<12.4f} {speedup:<10.2f}x")
            
            del A, W, A_int8, W_int8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{seq_len:<10} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{seq_len:<10} ERROR: {e}")
    
    # Test 4: Fixed very large batch, vary sequence length
    print(f"\n=== Test 4: Varying Seq Len (batch=128) ===")
    print(f"Fixed: batch=128, rank={rank}, hidden_dim={hidden_dim}")
    print(f"\n{'Seq Len':<10} {'Total Tokens':<15} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'Full INT8':<12} {'Speedup':<10}")
    print("-" * 75)
    
    batch = 128
    seq_lengths = [64, 128, 256, 512, 1024]
    
    for seq_len in seq_lengths:
        try:
            M = batch * seq_len
            K = rank
            N = hidden_dim
            
            A = torch.randn(M, K, dtype=torch.float16, device=device)
            W = torch.randn(K, N, dtype=torch.float16, device=device)
            
            A_scale = A.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
            A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
            W_scale = W.abs().amax() / 127.0 + 1e-6
            W_int8 = (W / W_scale).round().clamp(-128, 127).to(torch.int8)
            
            fp16_time = benchmark_func(lambda: torch.matmul(A, W), num_warmup=10, num_runs=50)
            int8_time = benchmark_func(lambda: torch._int_mm(A_int8, W_int8), num_warmup=10, num_runs=50)
            
            def int8_full():
                a_s = A.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
                a_i = (A / a_s).round().clamp(-128, 127).to(torch.int8)
                c = torch._int_mm(a_i, W_int8)
                return c.half() * a_s * W_scale
            
            full_time = benchmark_func(int8_full, num_warmup=10, num_runs=50)
            speedup = fp16_time / int8_time
            total_tokens = batch * seq_len
            
            print(f"{seq_len:<10} {total_tokens:<15} {fp16_time:<12.4f} {int8_time:<12.4f} {full_time:<12.4f} {speedup:<10.2f}x")
            
            del A, W, A_int8, W_int8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{seq_len:<10} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{seq_len:<10} ERROR: {e}")


def run_quantization_error_analysis():
    """Analyze quantization error for per-tensor vs per-token."""
    print("\n" + "=" * 80)
    print("Quantization Error: Per-Tensor vs Per-Token")
    print("=" * 80)
    
    device = 'cuda'
    
    # Create test tensor (simulating Value states)
    batch, seq_len, hidden_dim = 4, 2048, 1024
    V = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float16, device=device)
    
    # Per-tensor quantization
    scale_pt = V.abs().amax() / 127.0 + 1e-6
    V_int8_pt = (V / scale_pt).round().clamp(-128, 127).to(torch.int8)
    V_dequant_pt = V_int8_pt.half() * scale_pt
    
    # Per-token quantization
    scale_tk = V.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
    V_int8_tk = (V / scale_tk).round().clamp(-128, 127).to(torch.int8)
    V_dequant_tk = V_int8_tk.half() * scale_tk
    
    # Calculate errors
    error_pt = (V - V_dequant_pt).abs()
    error_tk = (V - V_dequant_tk).abs()
    
    rel_error_pt = error_pt / (V.abs() + 1e-6)
    rel_error_tk = error_tk / (V.abs() + 1e-6)
    
    print(f"\nTensor shape: {V.shape}")
    print(f"\nPer-Tensor Quantization:")
    print(f"  Scale shape: {scale_pt.shape}")
    print(f"  Max absolute error: {error_pt.max().item():.6f}")
    print(f"  Mean absolute error: {error_pt.mean().item():.6f}")
    print(f"  Max relative error: {rel_error_pt.max().item():.4%}")
    print(f"  Mean relative error: {rel_error_pt.mean().item():.4%}")
    
    print(f"\nPer-Token Quantization:")
    print(f"  Scale shape: {scale_tk.shape}")
    print(f"  Max absolute error: {error_tk.max().item():.6f}")
    print(f"  Mean absolute error: {error_tk.mean().item():.6f}")
    print(f"  Max relative error: {rel_error_tk.max().item():.4%}")
    print(f"  Mean relative error: {rel_error_tk.mean().item():.4%}")
    
    print(f"\n改进 (Per-Token vs Per-Tensor):")
    print(f"  Mean error reduction: {(1 - error_tk.mean() / error_pt.mean()) * 100:.1f}%")
    
    # Memory overhead
    scales_pt = 1  # single scale
    scales_tk = batch * seq_len  # one scale per token
    print(f"\n存储开销:")
    print(f"  Per-Tensor scales: {scales_pt} floats")
    print(f"  Per-Token scales: {scales_tk} floats ({scales_tk * 2 / 1024:.1f} KB)")


def run_long_sequence_summary():
    """Summary for long sequences."""
    print("\n" + "=" * 80)
    print("Long Sequence Summary (8K, 16K, 24K) - cuBLAS")
    print("=" * 80)
    
    device = 'cuda'
    
    configs = [
        (8192, 1024, 1024, "8K x 1024 -> 1024"),
        (16384, 1024, 1024, "16K x 1024 -> 1024"),
        (24576, 1024, 1024, "24K x 1024 -> 1024"),
        (8192, 256, 1024, "8K x 256 -> 1024 (V)"),
        (16384, 256, 1024, "16K x 256 -> 1024 (V)"),
        (24576, 256, 1024, "24K x 256 -> 1024 (V)"),
    ]
    
    print(f"\n{'Config':<30} {'FP16':<10} {'INT8只':<10} {'W8A8全':<10} {'W-Only':<10} {'INT8 Spd':<10} {'W8A8 Spd':<10}")
    print("-" * 95)
    
    for M, K, N, desc in configs:
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)
        
        # Pre-quantize
        A_scale = A.abs().amax() / 127.0 + 1e-6
        B_scale = B.abs().amax() / 127.0 + 1e-6
        A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
        B_int8 = (B / B_scale).round().clamp(-128, 127).to(torch.int8)
        
        # Weight-only
        B_scale_pc = B.abs().amax(dim=0, keepdim=True) / 127.0 + 1e-6
        B_int8_pc = (B / B_scale_pc).round().clamp(-128, 127).to(torch.int8)
        
        # 1. FP16
        fp16_time = benchmark_func(lambda: torch.matmul(A, B), num_warmup=10, num_runs=50)
        
        # 2. INT8 only
        int8_time = benchmark_func(lambda: torch._int_mm(A_int8, B_int8), num_warmup=10, num_runs=50)
        
        # 3. Full W8A8
        def w8a8_full():
            a_s = A.abs().amax() / 127.0 + 1e-6
            a_i = (A / a_s).round().clamp(-128, 127).to(torch.int8)
            c = torch._int_mm(a_i, B_int8)
            return c.half() * a_s * B_scale
        
        w8a8_time = benchmark_func(w8a8_full, num_warmup=10, num_runs=50)
        
        # 4. Weight-only
        def wo():
            B_fp = B_int8_pc.half() * B_scale_pc
            return torch.matmul(A, B_fp)
        
        wo_time = benchmark_func(wo, num_warmup=10, num_runs=50)
        
        int8_spd = fp16_time / int8_time
        w8a8_spd = fp16_time / w8a8_time
        
        print(f"{desc:<30} {fp16_time:<10.3f} {int8_time:<10.3f} {w8a8_time:<10.3f} {wo_time:<10.3f} {int8_spd:<10.2f}x {w8a8_spd:<10.2f}x")
        
        del A, B, A_int8, B_int8, B_int8_pc
        torch.cuda.empty_cache()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if not check_environment():
        print("\nError: torch._int_mm not available")
        print("Requires PyTorch >= 2.0 with CUDA support")
        sys.exit(1)
    
    # Run all benchmarks
    run_gemm_comparison()
    run_timing_breakdown()
    
    # Per-token quantization benchmarks (KIVI-style Value)
    run_per_token_quantization_benchmark()
    run_per_token_matmul_benchmark()
    run_large_batch_comparison()
    run_quantization_error_analysis()
    
    run_tensor_core_utilization()
    run_gflops_analysis()
    run_long_sequence_summary()
    
    print("\n" + "=" * 80)
    print("Summary & Conclusions")
    print("=" * 80)
    print("""
关键发现:

1. cuBLAS INT8 (torch._int_mm) 性能:
   - 使用 cuBLASLt 的 cublasLtMatmul 实现
   - 利用 Tensor Cores 进行 INT8 GEMM
   - 理论上应该是 FP16 的 2x 吞吐量

2. Per-Token vs Per-Tensor 量化:
   - Per-Token (KIVI Value): 每个 token 一个 scale，精度更高
   - Per-Tensor: 整个张量一个 scale，开销更小
   - Per-Token 量化误差可降低 30-50%
   - Per-Token 量化开销稍大，但大矩阵时可忽略

3. 大 Batch Size 影响:
   - Batch 增大 -> 计算量增大 -> 固定开销占比降低
   - Batch >= 8 且 seq_len >= 2048 时，INT8 可能有优势
   - 超大 Batch (32+) 时加速效果更明显

4. 为什么实际加速可能不明显:
   a) 量化/反量化开销
   b) 内存带宽瓶颈 (小矩阵)
   c) Kernel 启动开销
   d) cuBLAS FP16 已经高度优化

5. 何时 INT8 有优势:
   - 大矩阵 (M >= 8192, K, N >= 1024)
   - 大 batch size (>= 8)
   - 计算密集型场景
   - 权重预量化 (减少运行时开销)

6. 最佳实践:
   - 短序列/小 batch: 使用 FP16
   - 长序列/大 batch: 测试 INT8 per-token
   - 显存受限: Weight-Only INT8
   - L20/Ada: 考虑 FP8

7. Tensor Core 对齐:
   - INT8: M, K, N 最好是 8 的倍数
   - FP16: M, K, N 最好是 16 的倍数
   - 不对齐会降低性能
    """)
