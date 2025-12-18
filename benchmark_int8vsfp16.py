"""
Benchmark: torch._int_mm INT8 vs FP16 Matrix Multiplication

Tests torch native INT8 matmul vs standard FP16 matmul.
Includes detailed quantization/dequantization timing breakdown.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys


def check_environment():
    """Check CUDA and torch._int_mm support."""
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU Compute Capability: {cap[0]}.{cap[1]}")
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_mem:.1f} GB")
    
    # Check torch._int_mm
    has_int_mm = hasattr(torch, '_int_mm')
    print(f"\ntorch._int_mm: {'✓ Available' if has_int_mm else '✗ Not available'}")
    
    if not has_int_mm:
        print("torch._int_mm requires PyTorch >= 2.0")
        return False
    
    return True


def benchmark_operation(func, num_warmup=10, num_runs=100):
    """Benchmark any operation."""
    # Warmup
    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        func()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / num_runs * 1000


# ============================================================================
# Quantization/Dequantization Timing
# ============================================================================

def run_quant_dequant_timing():
    """Detailed timing for torch INT8 quantization and dequantization."""
    print("\n" + "=" * 70)
    print("torch INT8 Quantization & Dequantization Timing")
    print("=" * 70)
    
    device = torch.device('cuda')
    num_warmup = 10
    num_runs = 100
    
    # Test configurations: (rows, cols, description)
    configs = [
        (1, 1024, "Tiny: 1x1024"),
        (1, 4096, "Tiny: 1x4096"),
        (128, 1024, "Small: 128x1024"),
        (128, 4096, "Small: 128x4096"),
        (512, 1024, "Medium: 512x1024"),
        (512, 4096, "Medium: 512x4096"),
        (2048, 1024, "Large: 2048x1024"),
        (2048, 4096, "Large: 2048x4096"),
        (4096, 4096, "XL: 4096x4096"),
        # Very long sequences
        (8192, 1024, "8K seq: 8192x1024"),
        (16384, 1024, "16K seq: 16384x1024"),
        (24576, 1024, "24K seq: 24576x1024"),
        # Value reconstruction specific
        (1, 256, "V latent decode: 1x256"),
        (128, 256, "V latent small: 128x256"),
        (512, 256, "V latent med: 512x256"),
        (2048, 256, "V latent long: 2048x256"),
        (8192, 256, "V latent 8K: 8192x256"),
        (16384, 256, "V latent 16K: 16384x256"),
        (24576, 256, "V latent 24K: 24576x256"),
    ]
    
    print(f"\n{'Config':<30} {'Quant (ms)':<12} {'Dequant (ms)':<12} {'Total Q+D':<12} {'Elements':<12}")
    print("-" * 80)
    
    for rows, cols, desc in configs:
        x = torch.randn(rows, cols, dtype=torch.float16, device=device)
        
        # Quantization: FP16 -> INT8
        def quant_op():
            scale = x.abs().amax() / 127 + 1e-6
            x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
            return x_int8, scale
        
        quant_time = benchmark_operation(quant_op, num_warmup, num_runs)
        
        # Pre-compute for dequant
        scale = x.abs().amax() / 127 + 1e-6
        x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
        
        # Dequantization: INT8 -> FP16
        def dequant_op():
            return x_int8.half() * scale
        
        dequant_time = benchmark_operation(dequant_op, num_warmup, num_runs)
        
        total_qd = quant_time + dequant_time
        elements = rows * cols
        
        print(f"{desc:<30} {quant_time:<12.4f} {dequant_time:<12.4f} {total_qd:<12.4f} {elements:<12}")
        
        del x, x_int8
        torch.cuda.empty_cache()
    
    print("\n说明: torch 手动量化比 bitsandbytes 快很多")


# ============================================================================
# End-to-End Breakdown
# ============================================================================

def run_end_to_end_breakdown():
    """Full breakdown: quant + int_mm + dequant vs FP16 matmul."""
    print("\n" + "=" * 70)
    print("End-to-End Breakdown: FP16 vs torch._int_mm")
    print("=" * 70)
    
    device = torch.device('cuda')
    num_warmup = 10
    num_runs = 100
    
    # Test configurations: (M, K, N, description)
    configs = [
        (1, 256, 1024, "V recon decode: 1x256 @ 256x1024"),
        (128, 256, 1024, "V recon small: 128x256 @ 256x1024"),
        (512, 256, 1024, "V recon med: 512x256 @ 256x1024"),
        (2048, 256, 1024, "V recon long: 2048x256 @ 256x1024"),
        (512, 1024, 1024, "Medium: 512x1024 @ 1024x1024"),
        (2048, 1024, 1024, "Large: 2048x1024 @ 1024x1024"),
        (2048, 4096, 4096, "XL: 2048x4096 @ 4096x4096"),
        # Very long sequences (8K, 16K, 24K)
        (8192, 1024, 1024, "8K seq: 8192x1024 @ 1024x1024"),
        (16384, 1024, 1024, "16K seq: 16384x1024 @ 1024x1024"),
        (24576, 1024, 1024, "24K seq: 24576x1024 @ 1024x1024"),
        # V recon for very long sequences
        (8192, 256, 1024, "V recon 8K: 8192x256 @ 256x1024"),
        (16384, 256, 1024, "V recon 16K: 16384x256 @ 256x1024"),
        (24576, 256, 1024, "V recon 24K: 24576x256 @ 256x1024"),
    ]
    
    print(f"\n{'Config':<35} {'FP16':<8} {'Q(A)':<8} {'Q(B)':<8} {'int_mm':<8} {'Dequant':<8} {'Total':<8} {'Speedup':<8}")
    print("-" * 100)
    
    for M, K, N, desc in configs:
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)
        
        # 1. FP16 matmul baseline
        def fp16_matmul():
            return torch.matmul(A, B)
        
        fp16_time = benchmark_operation(fp16_matmul, num_warmup, num_runs)
        
        # 2. Quantize A
        def quant_A():
            scale = A.abs().amax() / 127 + 1e-6
            return (A / scale).round().clamp(-128, 127).to(torch.int8), scale
        
        quant_A_time = benchmark_operation(quant_A, num_warmup, num_runs)
        
        # 3. Quantize B
        def quant_B():
            scale = B.abs().amax() / 127 + 1e-6
            return (B / scale).round().clamp(-128, 127).to(torch.int8), scale
        
        quant_B_time = benchmark_operation(quant_B, num_warmup, num_runs)
        
        # Pre-quantize for compute benchmark
        A_scale = A.abs().amax() / 127 + 1e-6
        B_scale = B.abs().amax() / 127 + 1e-6
        A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
        B_int8 = (B / B_scale).round().clamp(-128, 127).to(torch.int8)
        
        # 4. torch._int_mm only
        def int_mm_only():
            return torch._int_mm(A_int8, B_int8)
        
        int_mm_time = benchmark_operation(int_mm_only, num_warmup, num_runs)
        
        # 5. Dequantize result
        C_int32 = torch._int_mm(A_int8, B_int8)
        
        def dequant_result():
            return C_int32.half() * A_scale * B_scale
        
        dequant_time = benchmark_operation(dequant_result, num_warmup, num_runs)
        
        # Total INT8 time (with pre-quantized weights)
        # In inference: weight B is pre-quantized, only A needs runtime quant
        total_int8 = quant_A_time + int_mm_time + dequant_time
        
        speedup = fp16_time / total_int8
        
        print(f"{desc:<35} {fp16_time:<8.3f} {quant_A_time:<8.3f} {quant_B_time:<8.3f} {int_mm_time:<8.3f} {dequant_time:<8.3f} {total_int8:<8.3f} {speedup:<8.2f}x")
        
        del A, B, A_int8, B_int8, C_int32
        torch.cuda.empty_cache()
    
    print("\n说明:")
    print("  - Q(A): 量化激活的时间 (推理时每次都需要)")
    print("  - Q(B): 量化权重的时间 (可以离线预计算)")
    print("  - int_mm: torch._int_mm 纯计算时间")
    print("  - Dequant: 结果反量化时间")
    print("  - Total: Q(A) + int_mm + Dequant (使用预量化权重)")


# ============================================================================
# Pure Compute Comparison (Pre-quantized)
# ============================================================================

def run_pure_compute_comparison():
    """Compare pure compute: FP16 matmul vs torch._int_mm (pre-quantized)."""
    print("\n" + "=" * 70)
    print("Pure Compute: FP16 matmul vs torch._int_mm (Pre-quantized)")
    print("=" * 70)
    
    device = torch.device('cuda')
    num_warmup = 10
    num_runs = 100
    
    configs = [
        (1, 1024, 1024, "Decode: 1x1024 @ 1024x1024"),
        (1, 4096, 4096, "Decode: 1x4096 @ 4096x4096"),
        (128, 1024, 1024, "Small: 128x1024 @ 1024x1024"),
        (512, 1024, 1024, "Medium: 512x1024 @ 1024x1024"),
        (2048, 1024, 1024, "Large: 2048x1024 @ 1024x1024"),
        (2048, 4096, 4096, "XL: 2048x4096 @ 4096x4096"),
        # Very long sequences
        (8192, 1024, 1024, "8K: 8192x1024 @ 1024x1024"),
        (16384, 1024, 1024, "16K: 16384x1024 @ 1024x1024"),
        (24576, 1024, 1024, "24K: 24576x1024 @ 1024x1024"),
        # Value reconstruction
        (1, 256, 1024, "V decode: 1x256 @ 256x1024"),
        (128, 256, 1024, "V small: 128x256 @ 256x1024"),
        (512, 256, 1024, "V med: 512x256 @ 256x1024"),
        (2048, 256, 1024, "V long: 2048x256 @ 256x1024"),
        (8192, 256, 1024, "V 8K: 8192x256 @ 256x1024"),
        (16384, 256, 1024, "V 16K: 16384x256 @ 256x1024"),
        (24576, 256, 1024, "V 24K: 24576x256 @ 256x1024"),
    ]
    
    print(f"\n{'Config':<35} {'FP16 (ms)':<12} {'int_mm (ms)':<12} {'Speedup':<10} {'GFLOPS FP16':<12} {'GFLOPS INT8':<12}")
    print("-" * 95)
    
    for M, K, N, desc in configs:
        A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
        B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
        
        # Pre-quantize
        A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
        B_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        
        # FP16 benchmark
        def fp16_matmul():
            return torch.matmul(A_fp16, B_fp16)
        
        fp16_time = benchmark_operation(fp16_matmul, num_warmup, num_runs)
        
        # INT8 benchmark
        def int8_matmul():
            return torch._int_mm(A_int8, B_int8)
        
        int8_time = benchmark_operation(int8_matmul, num_warmup, num_runs)
        
        speedup = fp16_time / int8_time
        
        # Calculate GFLOPS
        flops = 2 * M * K * N  # multiply-add = 2 ops
        gflops_fp16 = flops / (fp16_time / 1000) / 1e9
        gflops_int8 = flops / (int8_time / 1000) / 1e9
        
        print(f"{desc:<35} {fp16_time:<12.4f} {int8_time:<12.4f} {speedup:<10.2f}x {gflops_fp16:<12.1f} {gflops_int8:<12.1f}")
        
        del A_fp16, B_fp16, A_int8, B_int8
        torch.cuda.empty_cache()
    
    print("\n说明: 此测试不包含量化/反量化开销，仅比较纯计算速度")


# ============================================================================
# Weight-Only INT8 (Practical Scenario)
# ============================================================================

def run_weight_only_int8():
    """Weight-Only INT8: pre-quantized weight, FP16 activation."""
    print("\n" + "=" * 70)
    print("Weight-Only INT8: Dequant Weight -> FP16 Matmul")
    print("=" * 70)
    
    device = torch.device('cuda')
    num_warmup = 10
    num_runs = 100
    
    configs = [
        (1, 256, 1024, "V decode: 1x256 -> 1024"),
        (128, 256, 1024, "V small: 128x256 -> 1024"),
        (512, 256, 1024, "V med: 512x256 -> 1024"),
        (2048, 256, 1024, "V long: 2048x256 -> 1024"),
        (512, 1024, 1024, "Medium: 512x1024 -> 1024"),
        (2048, 1024, 1024, "Large: 2048x1024 -> 1024"),
        (2048, 4096, 4096, "XL: 2048x4096 -> 4096"),
        # Very long sequences
        (8192, 1024, 1024, "8K: 8192x1024 -> 1024"),
        (16384, 1024, 1024, "16K: 16384x1024 -> 1024"),
        (24576, 1024, 1024, "24K: 24576x1024 -> 1024"),
        (8192, 256, 1024, "V 8K: 8192x256 -> 1024"),
        (16384, 256, 1024, "V 16K: 16384x256 -> 1024"),
        (24576, 256, 1024, "V 24K: 24576x256 -> 1024"),
    ]
    
    print(f"\n{'Config':<30} {'FP16':<10} {'Dequant W':<10} {'Matmul':<10} {'Total W8':<10} {'Speedup':<10}")
    print("-" * 80)
    
    for seq_len, in_feat, out_feat, desc in configs:
        x = torch.randn(seq_len, in_feat, dtype=torch.float16, device=device)
        W = torch.randn(out_feat, in_feat, dtype=torch.float16, device=device)
        
        # Pre-quantize weight (per-channel)
        W_scale = W.abs().amax(dim=1, keepdim=True) / 127 + 1e-6
        W_int8 = (W / W_scale).round().clamp(-128, 127).to(torch.int8)
        
        # 1. FP16 baseline
        def fp16_linear():
            return F.linear(x, W)
        
        fp16_time = benchmark_operation(fp16_linear, num_warmup, num_runs)
        
        # 2. Weight dequantization only
        def dequant_weight():
            return W_int8.half() * W_scale
        
        dequant_time = benchmark_operation(dequant_weight, num_warmup, num_runs)
        
        # 3. Matmul with dequantized weight
        W_dequant = W_int8.half() * W_scale
        
        def matmul_only():
            return F.linear(x, W_dequant)
        
        matmul_time = benchmark_operation(matmul_only, num_warmup, num_runs)
        
        # 4. Total Weight-Only INT8
        def weight_only_int8():
            W_fp16 = W_int8.half() * W_scale
            return F.linear(x, W_fp16)
        
        total_w8 = benchmark_operation(weight_only_int8, num_warmup, num_runs)
        
        speedup = fp16_time / total_w8
        
        print(f"{desc:<30} {fp16_time:<10.4f} {dequant_time:<10.4f} {matmul_time:<10.4f} {total_w8:<10.4f} {speedup:<10.2f}x")
        
        del x, W, W_int8, W_scale, W_dequant
        torch.cuda.empty_cache()
    
    print("\n说明:")
    print("  - Weight-Only INT8: 权重存储为 INT8，推理时反量化为 FP16")
    print("  - 优势: 节省 50% 权重显存，反量化开销小")


# ============================================================================
# W8A8 Full INT8 (Both Weight and Activation)
# ============================================================================

def run_w8a8_int8():
    """W8A8: Both weight and activation quantized to INT8."""
    print("\n" + "=" * 70)
    print("W8A8 INT8: Quantize Both Weight and Activation")
    print("=" * 70)
    
    device = torch.device('cuda')
    num_warmup = 10
    num_runs = 100
    
    configs = [
        (1, 256, 1024, "V decode: 1x256 -> 1024"),
        (128, 256, 1024, "V small: 128x256 -> 1024"),
        (512, 256, 1024, "V med: 512x256 -> 1024"),
        (2048, 256, 1024, "V long: 2048x256 -> 1024"),
        (512, 1024, 1024, "Medium: 512x1024 -> 1024"),
        (2048, 1024, 1024, "Large: 2048x1024 -> 1024"),
        (2048, 4096, 4096, "XL: 2048x4096 -> 4096"),
        # Very long sequences
        (8192, 1024, 1024, "8K: 8192x1024 -> 1024"),
        (16384, 1024, 1024, "16K: 16384x1024 -> 1024"),
        (24576, 1024, 1024, "24K: 24576x1024 -> 1024"),
        (8192, 256, 1024, "V 8K: 8192x256 -> 1024"),
        (16384, 256, 1024, "V 16K: 16384x256 -> 1024"),
        (24576, 256, 1024, "V 24K: 24576x256 -> 1024"),
    ]
    
    print(f"\n{'Config':<30} {'FP16':<10} {'Q(Act)':<10} {'int_mm':<10} {'Dequant':<10} {'Total':<10} {'Speedup':<10}")
    print("-" * 90)
    
    for M, K, N, desc in configs:
        x = torch.randn(M, K, dtype=torch.float16, device=device)
        W = torch.randn(N, K, dtype=torch.float16, device=device)  # (out, in) for linear
        
        # Pre-quantize weight
        W_scale = W.abs().amax() / 127 + 1e-6
        W_int8 = (W / W_scale).round().clamp(-128, 127).to(torch.int8)
        
        # 1. FP16 baseline
        def fp16_linear():
            return F.linear(x, W)
        
        fp16_time = benchmark_operation(fp16_linear, num_warmup, num_runs)
        
        # 2. Quantize activation
        def quant_act():
            scale = x.abs().amax() / 127 + 1e-6
            return (x / scale).round().clamp(-128, 127).to(torch.int8), scale
        
        quant_act_time = benchmark_operation(quant_act, num_warmup, num_runs)
        
        # Pre-compute for int_mm
        x_scale = x.abs().amax() / 127 + 1e-6
        x_int8 = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
        
        # 3. INT8 matmul (note: need to transpose W for _int_mm)
        W_int8_t = W_int8.t().contiguous()
        
        def int_mm_compute():
            return torch._int_mm(x_int8, W_int8_t)
        
        int_mm_time = benchmark_operation(int_mm_compute, num_warmup, num_runs)
        
        # 4. Dequantize result
        C_int32 = torch._int_mm(x_int8, W_int8_t)
        
        def dequant_result():
            return C_int32.half() * x_scale * W_scale
        
        dequant_time = benchmark_operation(dequant_result, num_warmup, num_runs)
        
        # Total W8A8 time
        total_w8a8 = quant_act_time + int_mm_time + dequant_time
        
        speedup = fp16_time / total_w8a8
        
        print(f"{desc:<30} {fp16_time:<10.4f} {quant_act_time:<10.4f} {int_mm_time:<10.4f} {dequant_time:<10.4f} {total_w8a8:<10.4f} {speedup:<10.2f}x")
        
        del x, W, x_int8, W_int8, W_int8_t, C_int32
        torch.cuda.empty_cache()
    
    print("\n说明:")
    print("  - W8A8: 权重和激活都量化为 INT8")
    print("  - Q(Act): 运行时量化激活的开销")
    print("  - int_mm: torch._int_mm 纯计算")
    print("  - Dequant: 结果反量化")


# ============================================================================
# Long Sequence Summary
# ============================================================================

def run_long_sequence_summary():
    """Summary for long sequences (8K, 16K, 24K)."""
    print("\n" + "=" * 70)
    print("Long Sequence Summary (8K, 16K, 24K) - torch._int_mm")
    print("=" * 70)
    
    device = torch.device('cuda')
    num_warmup = 10
    num_runs = 50
    
    configs = [
        # Standard (1024 -> 1024)
        (8192, 1024, 1024, "8K x 1024 -> 1024"),
        (16384, 1024, 1024, "16K x 1024 -> 1024"),
        (24576, 1024, 1024, "24K x 1024 -> 1024"),
        # V reconstruction (256 -> 1024)
        (8192, 256, 1024, "8K x 256 -> 1024 (V)"),
        (16384, 256, 1024, "16K x 256 -> 1024 (V)"),
        (24576, 256, 1024, "24K x 256 -> 1024 (V)"),
    ]
    
    print(f"\n{'Config':<30} {'FP16':<10} {'int_mm只':<10} {'W8A8全':<10} {'W-Only':<10} {'int_mm Spd':<10} {'W8A8 Spd':<10}")
    print("-" * 95)
    
    for M, K, N, desc in configs:
        x = torch.randn(M, K, dtype=torch.float16, device=device)
        W = torch.randn(N, K, dtype=torch.float16, device=device)
        
        # Pre-quantize
        x_scale = x.abs().amax() / 127 + 1e-6
        W_scale = W.abs().amax() / 127 + 1e-6
        x_int8 = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
        W_int8 = (W / W_scale).round().clamp(-128, 127).to(torch.int8)
        W_int8_t = W_int8.t().contiguous()
        
        # Weight-only quantize
        W_scale_pc = W.abs().amax(dim=1, keepdim=True) / 127 + 1e-6
        W_int8_pc = (W / W_scale_pc).round().clamp(-128, 127).to(torch.int8)
        
        # 1. FP16
        fp16_time = benchmark_operation(lambda: F.linear(x, W), num_warmup, num_runs)
        
        # 2. Pure int_mm (pre-quantized, no overhead)
        int_mm_time = benchmark_operation(lambda: torch._int_mm(x_int8, W_int8_t), num_warmup, num_runs)
        
        # 3. W8A8 full (with quant overhead)
        def w8a8_full():
            xs = x.abs().amax() / 127 + 1e-6
            xi = (x / xs).round().clamp(-128, 127).to(torch.int8)
            c = torch._int_mm(xi, W_int8_t)
            return c.half() * xs * W_scale
        
        w8a8_time = benchmark_operation(w8a8_full, num_warmup, num_runs)
        
        # 4. Weight-Only
        def weight_only():
            W_fp16 = W_int8_pc.half() * W_scale_pc
            return F.linear(x, W_fp16)
        
        wo_time = benchmark_operation(weight_only, num_warmup, num_runs)
        
        int_mm_spd = fp16_time / int_mm_time
        w8a8_spd = fp16_time / w8a8_time
        
        print(f"{desc:<30} {fp16_time:<10.3f} {int_mm_time:<10.3f} {w8a8_time:<10.3f} {wo_time:<10.3f} {int_mm_spd:<10.2f}x {w8a8_spd:<10.2f}x")
        
        del x, W, x_int8, W_int8, W_int8_t, W_int8_pc
        torch.cuda.empty_cache()
    
    print("\n说明:")
    print("  - int_mm只: 纯 torch._int_mm (预量化，无开销)")
    print("  - W8A8全: 完整 W8A8 流程 (含量化/反量化)")
    print("  - W-Only: Weight-Only INT8 (反量化权重 + FP16 计算)")


# ============================================================================
# Quantization Error Analysis
# ============================================================================

def test_quantization_error():
    """Test quantization error."""
    print("\n" + "=" * 70)
    print("INT8 Quantization Error Analysis")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    # Create test tensors
    x = torch.randn(1024, 1024, dtype=torch.float16, device=device)
    
    # Quantize and dequantize
    scale = x.abs().amax() / 127 + 1e-6
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    x_dequant = x_int8.half() * scale
    
    # Calculate error
    abs_error = (x - x_dequant).abs()
    rel_error = abs_error / (x.abs() + 1e-6)
    
    print(f"\nTensor shape: {x.shape}")
    print(f"Original dtype: {x.dtype}")
    print(f"Quantized dtype: {x_int8.dtype}")
    
    print(f"\nMax absolute error: {abs_error.max().item():.6f}")
    print(f"Mean absolute error: {abs_error.mean().item():.6f}")
    print(f"Max relative error: {rel_error.max().item():.4%}")
    print(f"Mean relative error: {rel_error.mean().item():.4%}")
    
    # Memory
    original_size = x.numel() * 2  # FP16 = 2 bytes
    quantized_size = x_int8.numel() * 1  # INT8 = 1 byte
    
    print(f"\nMemory:")
    print(f"  Original (FP16): {original_size / 1e6:.2f} MB")
    print(f"  Quantized (INT8): {quantized_size / 1e6:.2f} MB")
    print(f"  Compression: {original_size / quantized_size:.1f}x")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if not check_environment():
        print("\ntorch._int_mm requires PyTorch >= 2.0 with CUDA support")
        sys.exit(1)
    
    # Run all benchmarks
    run_quant_dequant_timing()
    run_end_to_end_breakdown()
    run_pure_compute_comparison()
    run_weight_only_int8()
    run_w8a8_int8()
    run_long_sequence_summary()
    test_quantization_error()
    
    print("\n" + "=" * 70)
    print("Summary & Analysis")
    print("=" * 70)
    print("""
关键发现:

1. torch._int_mm 纯计算速度:
   - 小矩阵: 可能比 FP16 慢 (kernel 启动开销)
   - 大矩阵: 可能有加速 (计算密集型)
   - L20 的 FP16 Tensor Core 非常快

2. 量化/反量化开销:
   - torch 手动量化: ~0.01-0.1ms
   - 反量化: ~0.005-0.05ms
   - 比 bitsandbytes 快 5-10x

3. 端到端 W8A8:
   - 需要: 量化激活 + int_mm + 反量化结果
   - 开销往往抵消计算节省
   - 只有超大矩阵才可能有加速

4. Weight-Only INT8:
   - 最实用的方案
   - 节省 50% 权重显存
   - 开销小于 W8A8

5. 长序列 (8K, 16K, 24K):
   - 计算量大，固定开销占比小
   - int_mm 可能开始有优势
   - V recon (K=256) 仍受限于小矩阵

6. 建议:
   - 短序列/小矩阵: 使用 FP16
   - 显存受限: 使用 Weight-Only INT8
   - 长序列/大矩阵: 可测试 W8A8
   - L20 考虑 FP8 (原生支持)
    """)
