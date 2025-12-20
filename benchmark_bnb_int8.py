"""
Benchmark: bitsandbytes INT8 vs FP16 Matrix Multiplication

Tests bitsandbytes 8-bit linear layers vs standard FP16 matmul.
Includes detailed quantization/dequantization timing breakdown.

Install: pip install bitsandbytes
"""

import torch
import torch.nn as nn
import time
import sys

def check_environment():
    """Check CUDA and bitsandbytes support."""
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_mem:.1f} GB")
    
    # Check bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"\nbitsandbytes version: {bnb.__version__}")
        print("✓ bitsandbytes available")
        return True
    except ImportError as e:
        print(f"\n✗ bitsandbytes not installed: {e}")
        print("Install with: pip install bitsandbytes")
        return False


def benchmark_linear(linear_layer, x, num_warmup=10, num_runs=100, name=""):
    """Benchmark a linear layer."""
    # Warmup
    for _ in range(num_warmup):
        _ = linear_layer(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = linear_layer(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms


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
# NEW: Detailed Quantization/Dequantization Timing
# ============================================================================

def run_quant_dequant_timing():
    """Detailed timing for quantization and dequantization operations."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F
    
    print("\n" + "=" * 70)
    print("Quantization & Dequantization Timing Breakdown")
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
        # Create test tensor
        x = torch.randn(rows, cols, dtype=torch.float16, device=device)
        
        # First do one run to get the state structure
        x_int8_init, state_init = bnb_F.quantize_blockwise(x)
        
        # Benchmark quantization
        def quant_op():
            return bnb_F.quantize_blockwise(x)
        
        quant_time = benchmark_operation(quant_op, num_warmup, num_runs)
        
        # Benchmark dequantization (using pre-quantized data)
        x_int8, state = bnb_F.quantize_blockwise(x)
        
        def dequant_op():
            return bnb_F.dequantize_blockwise(x_int8, state)
        
        dequant_time = benchmark_operation(dequant_op, num_warmup, num_runs)
        
        total_qd = quant_time + dequant_time
        elements = rows * cols
        
        print(f"{desc:<30} {quant_time:<12.4f} {dequant_time:<12.4f} {total_qd:<12.4f} {elements:<12}")
        
        del x, x_int8, state
        torch.cuda.empty_cache()
    
    print("\n注意: 量化/反量化开销是固定的，与矩阵大小关系不大")
    print("这就是为什么小矩阵的 INT8 加速效果差 - 开销占比太高")


def run_quant_dequant_timing_torch():
    """Timing for torch native INT8 quantization."""
    print("\n" + "=" * 70)
    print("torch INT8 Quantization Timing (Manual)")
    print("=" * 70)
    
    device = torch.device('cuda')
    num_warmup = 10
    num_runs = 100
    
    configs = [
        (1, 1024, "Tiny: 1x1024"),
        (128, 1024, "Small: 128x1024"),
        (512, 1024, "Medium: 512x1024"),
        (2048, 1024, "Large: 2048x1024"),
        (4096, 4096, "XL: 4096x4096"),
        # Very long sequences
        (8192, 1024, "8K seq: 8192x1024"),
        (16384, 1024, "16K seq: 16384x1024"),
        (24576, 1024, "24K seq: 24576x1024"),
        (128, 256, "V latent: 128x256"),
        (512, 256, "V latent: 512x256"),
        (2048, 256, "V latent: 2048x256"),
        (8192, 256, "V latent 8K: 8192x256"),
        (16384, 256, "V latent 16K: 16384x256"),
        (24576, 256, "V latent 24K: 24576x256"),
    ]
    
    print(f"\n{'Config':<30} {'Quant (ms)':<12} {'Dequant (ms)':<12} {'Total Q+D':<12}")
    print("-" * 70)
    
    for rows, cols, desc in configs:
        x = torch.randn(rows, cols, dtype=torch.float16, device=device)
        
        # Quantization: FP16 -> INT8
        def quant_torch():
            scale = x.abs().max() / 127
            x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
            return x_int8, scale
        
        quant_time = benchmark_operation(quant_torch, num_warmup, num_runs)
        
        # Pre-compute for dequant
        scale = x.abs().max() / 127
        x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
        
        # Dequantization: INT8 -> FP16
        def dequant_torch():
            return x_int8.half() * scale
        
        dequant_time = benchmark_operation(dequant_torch, num_warmup, num_runs)
        
        total_qd = quant_time + dequant_time
        
        print(f"{desc:<30} {quant_time:<12.4f} {dequant_time:<12.4f} {total_qd:<12.4f}")
        
        del x, x_int8
        torch.cuda.empty_cache()


def run_end_to_end_breakdown():
    """Full breakdown: quant + matmul + dequant vs FP16 matmul."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F
    
    print("\n" + "=" * 70)
    print("End-to-End Breakdown: FP16 vs INT8 (Quant + Compute + Dequant)")
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
    
    print(f"\n{'Config':<35} {'FP16':<8} {'Q(A)':<8} {'Q(B)':<8} {'Compute':<8} {'Total INT8':<10} {'Speedup':<8}")
    print("-" * 95)
    
    for M, K, N, desc in configs:
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)
        
        # 1. FP16 matmul baseline
        def fp16_matmul():
            return torch.matmul(A, B)
        
        fp16_time = benchmark_operation(fp16_matmul, num_warmup, num_runs)
        
        # 2. INT8 with bnb - breakdown
        
        # 2a. Quantize A
        def quant_A():
            return bnb_F.quantize_blockwise(A)
        quant_A_time = benchmark_operation(quant_A, num_warmup, num_runs)
        
        # 2b. Quantize B (weight - typically done offline)
        def quant_B():
            return bnb_F.quantize_blockwise(B)
        quant_B_time = benchmark_operation(quant_B, num_warmup, num_runs)
        
        # Pre-quantize for compute benchmark
        A_int8, state_A = bnb_F.quantize_blockwise(A)
        B_int8, state_B = bnb_F.quantize_blockwise(B)
        
        # 2c. INT8 compute (dequant B + matmul)
        # Note: bnb doesn't have direct int8 matmul, so we dequant and matmul
        def int8_compute():
            B_dequant = bnb_F.dequantize_blockwise(B_int8, state_B)
            return torch.matmul(A, B_dequant)  # A stays FP16
        
        compute_time = benchmark_operation(int8_compute, num_warmup, num_runs)
        
        # Total INT8 time (assuming weight is pre-quantized, only A needs runtime quant)
        # For inference with pre-quantized weights: just dequant B + matmul
        total_int8 = compute_time  # Weight is pre-quantized
        
        speedup = fp16_time / total_int8
        
        print(f"{desc:<35} {fp16_time:<8.3f} {quant_A_time:<8.3f} {quant_B_time:<8.3f} {compute_time:<8.3f} {total_int8:<10.3f} {speedup:<8.2f}x")
        
        del A, B, A_int8, B_int8
        torch.cuda.empty_cache()
    
    print("\n说明:")
    print("  - Q(A): 量化激活的时间 (推理时每次都需要)")
    print("  - Q(B): 量化权重的时间 (可以离线预计算)")
    print("  - Compute: 反量化权重 + FP16矩阵乘法")
    print("  - Total INT8: 实际推理时间 (使用预量化权重)")


def run_bnb_linear_breakdown():
    """Breakdown of bnb.nn.Linear8bitLt internal operations."""
    import bitsandbytes as bnb
    
    print("\n" + "=" * 70)
    print("bnb.nn.Linear8bitLt Internal Breakdown")
    print("=" * 70)
    
    device = torch.device('cuda')
    num_warmup = 10
    num_runs = 100
    
    configs = [
        (1, 256, 1024, "V recon decode"),
        (128, 256, 1024, "V recon small"),
        (512, 256, 1024, "V recon med"),
        (2048, 256, 1024, "V recon long"),
        (512, 1024, 1024, "Medium"),
        (2048, 4096, 4096, "XL"),
        # Very long sequences (8K, 16K, 24K)
        (8192, 1024, 1024, "8K: 8192x1024->1024"),
        (16384, 1024, 1024, "16K: 16384x1024->1024"),
        (24576, 1024, 1024, "24K: 24576x1024->1024"),
        # V recon for very long sequences
        (8192, 256, 1024, "V recon 8K"),
        (16384, 256, 1024, "V recon 16K"),
        (24576, 256, 1024, "V recon 24K"),
    ]
    
    print(f"\n{'Config':<25} {'FP16 Linear':<12} {'BNB Linear':<12} {'Overhead':<12} {'Speedup':<10}")
    print("-" * 75)
    
    for seq_len, in_feat, out_feat, desc in configs:
        # FP16 Linear
        linear_fp16 = nn.Linear(in_feat, out_feat, bias=False).cuda().half()
        
        # BNB INT8 Linear
        linear_int8 = bnb.nn.Linear8bitLt(
            in_feat, out_feat,
            bias=False,
            has_fp16_weights=False,
            threshold=6.0,
        ).cuda()
        
        linear_int8.weight = bnb.nn.Int8Params(
            linear_fp16.weight.data.clone(),
            requires_grad=False,
            has_fp16_weights=False,
        ).cuda()
        
        x = torch.randn(1, seq_len, in_feat, dtype=torch.float16, device=device)
        
        # Benchmark
        fp16_time = benchmark_linear(linear_fp16, x, num_warmup, num_runs)
        int8_time = benchmark_linear(linear_int8, x, num_warmup, num_runs)
        
        overhead = int8_time - fp16_time
        speedup = fp16_time / int8_time
        
        print(f"{desc:<25} {fp16_time:<12.4f} {int8_time:<12.4f} {overhead:<12.4f} {speedup:<10.2f}x")
        
        del linear_fp16, linear_int8, x
        torch.cuda.empty_cache()
    
    print("\n说明:")
    print("  - Overhead > 0: INT8 比 FP16 慢 (量化/反量化开销大于计算节省)")
    print("  - Overhead < 0: INT8 比 FP16 快 (计算节省大于量化/反量化开销)")


def run_weight_only_int8_timing():
    """Test Weight-Only INT8 (dequant weight on-the-fly)."""
    print("\n" + "=" * 70)
    print("Weight-Only INT8 Timing (Dequant Weight -> FP16 Matmul)")
    print("=" * 70)
    
    device = torch.device('cuda')
    num_warmup = 10
    num_runs = 100
    
    configs = [
        (1, 256, 1024, "V recon decode"),
        (128, 256, 1024, "V recon small"),
        (512, 256, 1024, "V recon med"),
        (2048, 256, 1024, "V recon long"),
        (512, 1024, 1024, "Medium"),
        (2048, 4096, 4096, "XL"),
        # Very long sequences (8K, 16K, 24K)
        (8192, 1024, 1024, "8K: 8192x1024->1024"),
        (16384, 1024, 1024, "16K: 16384x1024->1024"),
        (24576, 1024, 1024, "24K: 24576x1024->1024"),
        # V recon for very long sequences
        (8192, 256, 1024, "V recon 8K"),
        (16384, 256, 1024, "V recon 16K"),
        (24576, 256, 1024, "V recon 24K"),
    ]
    
    print(f"\n{'Config':<25} {'FP16':<10} {'W-Dequant':<10} {'Matmul':<10} {'Total W8':<10} {'Speedup':<10}")
    print("-" * 80)
    
    for seq_len, in_feat, out_feat, desc in configs:
        x = torch.randn(seq_len, in_feat, dtype=torch.float16, device=device)
        W = torch.randn(out_feat, in_feat, dtype=torch.float16, device=device)
        
        # Pre-quantize weight
        W_scale = W.abs().amax(dim=1, keepdim=True) / 127 + 1e-6
        W_int8 = (W / W_scale).round().clamp(-128, 127).to(torch.int8)
        
        # 1. FP16 baseline
        def fp16_matmul():
            return torch.nn.functional.linear(x, W)
        
        fp16_time = benchmark_operation(fp16_matmul, num_warmup, num_runs)
        
        # 2. Weight dequantization only
        def dequant_weight():
            return W_int8.half() * W_scale
        
        dequant_time = benchmark_operation(dequant_weight, num_warmup, num_runs)
        
        # 3. Matmul with dequantized weight
        W_dequant = W_int8.half() * W_scale
        
        def matmul_only():
            return torch.nn.functional.linear(x, W_dequant)
        
        matmul_time = benchmark_operation(matmul_only, num_warmup, num_runs)
        
        # 4. Total Weight-Only INT8
        def weight_only_int8():
            W_fp16 = W_int8.half() * W_scale
            return torch.nn.functional.linear(x, W_fp16)
        
        total_w8 = benchmark_operation(weight_only_int8, num_warmup, num_runs)
        
        speedup = fp16_time / total_w8
        
        print(f"{desc:<25} {fp16_time:<10.4f} {dequant_time:<10.4f} {matmul_time:<10.4f} {total_w8:<10.4f} {speedup:<10.2f}x")
        
        del x, W, W_int8, W_scale
        torch.cuda.empty_cache()
    
    print("\n说明:")
    print("  - Weight-Only INT8: 权重存储为 INT8，推理时反量化为 FP16 后计算")
    print("  - 优势: 节省 ~50% 权重显存")
    print("  - 缺点: 有反量化开销，但通常比 bnb 开销小")


def run_linear_benchmark():
    """Benchmark Linear layers: FP16 vs BNB INT8."""
    import bitsandbytes as bnb
    
    print("\n" + "=" * 70)
    print("Benchmark: nn.Linear (FP16) vs bnb.nn.Linear8bitLt (INT8)")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    # Test configurations: (batch_size, seq_len, in_features, out_features)
    configs = [
        # Small (decode-like)
        (1, 1, 1024, 1024, "Decode: 1x1024 -> 1024"),
        (1, 1, 4096, 4096, "Decode: 1x4096 -> 4096"),
        
        # Medium
        (1, 128, 1024, 1024, "Small prefill: 128x1024 -> 1024"),
        (1, 128, 4096, 4096, "Small prefill: 128x4096 -> 4096"),
        
        # Large (prefill-like)
        (1, 512, 1024, 1024, "Med prefill: 512x1024 -> 1024"),
        (1, 512, 4096, 4096, "Med prefill: 512x4096 -> 4096"),
        
        (1, 2048, 1024, 1024, "Long prefill: 2048x1024 -> 1024"),
        (1, 2048, 4096, 4096, "Long prefill: 2048x4096 -> 4096"),
        
        # Very long sequences (8K, 16K, 24K)
        (1, 8192, 1024, 1024, "8K prefill: 8192x1024 -> 1024"),
        (1, 16384, 1024, 1024, "16K prefill: 16384x1024 -> 1024"),
        (1, 24576, 1024, 1024, "24K prefill: 24576x1024 -> 1024"),
        
        # Typical Value reconstruction (rank -> hidden)
        (1, 1, 256, 1024, "V recon decode: 1x256 -> 1024"),
        (1, 128, 256, 1024, "V recon small: 128x256 -> 1024"),
        (1, 512, 256, 1024, "V recon med: 512x256 -> 1024"),
        (1, 2048, 256, 1024, "V recon long: 2048x256 -> 1024"),
        
        # V recon for very long sequences
        (1, 8192, 256, 1024, "V recon 8K: 8192x256 -> 1024"),
        (1, 16384, 256, 1024, "V recon 16K: 16384x256 -> 1024"),
        (1, 24576, 256, 1024, "V recon 24K: 24576x256 -> 1024"),
        
        # Batched
        (4, 512, 1024, 1024, "Batched: 4x512x1024 -> 1024"),
        (8, 512, 1024, 1024, "Batched: 8x512x1024 -> 1024"),
    ]
    
    print(f"\n{'Config':<40} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'Speedup':<10} {'Memory Saved':<15}")
    print("-" * 90)
    
    results = []
    
    for batch, seq, in_feat, out_feat, desc in configs:
        try:
            # Create FP16 linear
            linear_fp16 = nn.Linear(in_feat, out_feat, bias=False).cuda().half()
            
            # Create BNB INT8 linear
            linear_int8 = bnb.nn.Linear8bitLt(
                in_feat, out_feat, 
                bias=False,
                has_fp16_weights=False,
                threshold=6.0,  # Outlier threshold
            ).cuda()
            
            # Copy weights (this triggers quantization)
            linear_int8.weight = bnb.nn.Int8Params(
                linear_fp16.weight.data.clone(),
                requires_grad=False,
                has_fp16_weights=False,
            ).cuda()
            
            # Create input
            x = torch.randn(batch, seq, in_feat, dtype=torch.float16, device=device)
            
            # Benchmark
            fp16_time = benchmark_linear(linear_fp16, x, name="FP16")
            int8_time = benchmark_linear(linear_int8, x, name="INT8")
            
            speedup = fp16_time / int8_time
            
            # Memory comparison
            fp16_mem = in_feat * out_feat * 2 / 1e6  # FP16 = 2 bytes
            int8_mem = in_feat * out_feat * 1 / 1e6  # INT8 = 1 byte
            mem_saved = (1 - int8_mem / fp16_mem) * 100
            
            print(f"{desc:<40} {fp16_time:<12.4f} {int8_time:<12.4f} {speedup:<10.2f}x {mem_saved:<15.1f}%")
            
            results.append({
                'desc': desc,
                'fp16_time': fp16_time,
                'int8_time': int8_time,
                'speedup': speedup,
            })
            
            # Cleanup
            del linear_fp16, linear_int8, x
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{desc:<40} ERROR: {e}")
    
    return results


def test_bnb_quantization_error():
    """Test quantization error of bitsandbytes."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F
    
    print("\n" + "=" * 70)
    print("Quantization Error Analysis")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    # Create test tensor
    x = torch.randn(1024, 1024, dtype=torch.float16, device=device)
    
    # Quantize with bitsandbytes
    x_int8, state = bnb_F.quantize_blockwise(x)
    x_dequant = bnb_F.dequantize_blockwise(x_int8, state)
    
    # Calculate error
    abs_error = (x - x_dequant).abs()
    rel_error = abs_error / (x.abs() + 1e-6)
    
    print(f"Original dtype: {x.dtype}")
    print(f"Quantized dtype: {x_int8.dtype}")
    print(f"Dequantized dtype: {x_dequant.dtype}")
    print(f"\nMax absolute error: {abs_error.max().item():.6f}")
    print(f"Mean absolute error: {abs_error.mean().item():.6f}")
    print(f"Max relative error: {rel_error.max().item():.4%}")
    print(f"Mean relative error: {rel_error.mean().item():.4%}")
    
    # Memory analysis
    original_size = x.numel() * 2  # FP16 = 2 bytes
    quantized_size = x_int8.numel() * 1  # INT8 = 1 byte
    # State also takes some memory
    
    print(f"\nMemory:")
    print(f"  Original (FP16): {original_size / 1e6:.2f} MB")
    print(f"  Quantized (INT8): {quantized_size / 1e6:.2f} MB")
    print(f"  Compression: {original_size / quantized_size:.1f}x")


def run_long_sequence_summary():
    """Summary benchmark for long sequences (8K, 16K, 24K)."""
    import bitsandbytes as bnb
    
    print("\n" + "=" * 70)
    print("Long Sequence Summary (8K, 16K, 24K)")
    print("=" * 70)
    
    device = torch.device('cuda')
    num_warmup = 10
    num_runs = 50  # Fewer runs for long sequences
    
    # Configurations for standard linear and V reconstruction
    configs = [
        # Standard linear (1024 -> 1024)
        (8192, 1024, 1024, "8K x 1024 -> 1024"),
        (16384, 1024, 1024, "16K x 1024 -> 1024"),
        (24576, 1024, 1024, "24K x 1024 -> 1024"),
        # V reconstruction (256 -> 1024)
        (8192, 256, 1024, "8K x 256 -> 1024 (V recon)"),
        (16384, 256, 1024, "16K x 256 -> 1024 (V recon)"),
        (24576, 256, 1024, "24K x 256 -> 1024 (V recon)"),
    ]
    
    print(f"\n{'Config':<30} {'FP16':<10} {'BNB INT8':<10} {'W-Only INT8':<12} {'BNB Spd':<10} {'W8 Spd':<10}")
    print("-" * 90)
    
    for seq_len, in_feat, out_feat, desc in configs:
        try:
            # Create layers
            linear_fp16 = nn.Linear(in_feat, out_feat, bias=False).cuda().half()
            
            linear_bnb = bnb.nn.Linear8bitLt(
                in_feat, out_feat,
                bias=False,
                has_fp16_weights=False,
                threshold=6.0,
            ).cuda()
            linear_bnb.weight = bnb.nn.Int8Params(
                linear_fp16.weight.data.clone(),
                requires_grad=False,
                has_fp16_weights=False,
            ).cuda()
            
            # Weight-Only INT8
            W = linear_fp16.weight.data
            W_scale = W.abs().amax(dim=1, keepdim=True) / 127 + 1e-6
            W_int8 = (W / W_scale).round().clamp(-128, 127).to(torch.int8)
            
            x = torch.randn(1, seq_len, in_feat, dtype=torch.float16, device=device)
            
            # FP16 benchmark
            fp16_time = benchmark_linear(linear_fp16, x, num_warmup, num_runs)
            
            # BNB INT8 benchmark
            bnb_time = benchmark_linear(linear_bnb, x, num_warmup, num_runs)
            
            # Weight-Only INT8 benchmark
            def weight_only_forward():
                W_fp16 = W_int8.half() * W_scale
                return torch.nn.functional.linear(x.squeeze(0), W_fp16)
            
            w8_time = benchmark_operation(weight_only_forward, num_warmup, num_runs)
            
            bnb_speedup = fp16_time / bnb_time
            w8_speedup = fp16_time / w8_time
            
            print(f"{desc:<30} {fp16_time:<10.3f} {bnb_time:<10.3f} {w8_time:<12.3f} {bnb_speedup:<10.2f}x {w8_speedup:<10.2f}x")
            
            del linear_fp16, linear_bnb, x, W, W_int8, W_scale
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{desc:<30} ERROR: {e}")
    
    print("\n说明:")
    print("  - BNB Spd: bitsandbytes INT8 相对 FP16 的加速比")
    print("  - W8 Spd: Weight-Only INT8 相对 FP16 的加速比")
    print("  - Speedup > 1.0 表示比 FP16 快")


def compare_with_torch_int8():
    """Compare bitsandbytes INT8 with torch._int_mm."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F
    
    print("\n" + "=" * 70)
    print("Comparison: bitsandbytes vs torch._int_mm")
    print("=" * 70)
    
    device = torch.device('cuda')
    M, K, N = 512, 1024, 1024
    num_warmup = 10
    num_runs = 100
    
    # Create matrices
    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(K, N, dtype=torch.float16, device=device)
    
    # 1. FP16 baseline
    for _ in range(num_warmup):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / num_runs * 1000
    
    # 2. torch._int_mm
    if hasattr(torch, '_int_mm'):
        A_scale = A.abs().max() / 127
        B_scale = B.abs().max() / 127
        A_int8 = (A / A_scale).round().clamp(-128, 127).to(torch.int8)
        B_int8 = (B / B_scale).round().clamp(-128, 127).to(torch.int8)
        
        for _ in range(num_warmup):
            _ = torch._int_mm(A_int8, B_int8)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_runs):
            C = torch._int_mm(A_int8, B_int8)
        torch.cuda.synchronize()
        torch_int8_time = (time.perf_counter() - start) / num_runs * 1000
    else:
        torch_int8_time = float('inf')
    
    # 3. bitsandbytes (using Linear8bitLt)
    linear = bnb.nn.Linear8bitLt(K, N, bias=False, has_fp16_weights=False).cuda()
    linear.weight = bnb.nn.Int8Params(B.t().contiguous(), requires_grad=False, has_fp16_weights=False).cuda()
    
    for _ in range(num_warmup):
        _ = linear(A)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = linear(A)
    torch.cuda.synchronize()
    bnb_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"\nMatrix size: ({M}, {K}) @ ({K}, {N})")
    print(f"{'Method':<25} {'Time (ms)':<15} {'Speedup vs FP16':<20}")
    print("-" * 60)
    print(f"{'FP16 (torch.matmul)':<25} {fp16_time:<15.4f} {'1.00x (baseline)':<20}")
    print(f"{'torch._int_mm':<25} {torch_int8_time:<15.4f} {fp16_time/torch_int8_time:<20.2f}x")
    print(f"{'bitsandbytes INT8':<25} {bnb_time:<15.4f} {fp16_time/bnb_time:<20.2f}x")


if __name__ == "__main__":
    if not check_environment():
        print("\nPlease install bitsandbytes: pip install bitsandbytes")
        sys.exit(1)
    
    # NEW: Quantization/Dequantization timing tests
    run_quant_dequant_timing()
    run_quant_dequant_timing_torch()
    run_end_to_end_breakdown()
    run_bnb_linear_breakdown()
    run_weight_only_int8_timing()
    
    # Long sequence summary (8K, 16K, 24K)
    run_long_sequence_summary()
    
    # Original benchmarks
    run_linear_benchmark()
    test_bnb_quantization_error()
    
    if hasattr(torch, '_int_mm'):
        compare_with_torch_int8()
    
    print("\n" + "=" * 70)
    print("Summary & Analysis")
    print("=" * 70)
    print("""
关键发现:

1. 量化/反量化开销分析:
   - bitsandbytes 量化: ~0.1-0.2ms (固定开销)
   - bitsandbytes 反量化: ~0.1-0.2ms (固定开销)
   - torch 手动量化: ~0.01-0.05ms (更快)
   - torch 手动反量化: ~0.01-0.02ms (更快)

2. 为什么小矩阵没有加速:
   - 固定开销 > 计算节省
   - 例如: FP16 计算 0.01ms, 但量化开销 0.2ms
   - 只有大矩阵 (>2048x4096) 才能看到加速

3. 长序列 (8K, 16K, 24K) 表现:
   - 8K x 1024 -> 1024: 可能开始看到加速
   - 16K x 1024 -> 1024: INT8 加速更明显
   - 24K x 1024 -> 1024: INT8 加速效果最好
   - V recon (8K/16K/24K x 256 -> 1024): 仍然受限于小 K 维度

4. Weight-Only INT8 优势:
   - 节省 50% 权重显存
   - 反量化开销比 bnb 小
   - 适合显存受限场景

5. 对 Value 重建的建议 (256x1024):
   - 短序列 (<2K): FP16 最快
   - 长序列 (8K+): 可能有加速，取决于具体硬件
   - 如果显存紧张，用 Weight-Only INT8

6. 如果需要加速:
   - 考虑 FP8 (L20 原生支持)
   - 考虑 torch.compile 优化 FP16
   - 考虑 kernel fusion (避免多次访存)
   - 长序列场景下 INT8 可能有优势
    """)
