"""
FP8 vs FP16 矩阵乘法性能对比

FP8 在 Ada Lovelace (L20, RTX 40xx) 上原生支持
- 输出直接是 FP16，避免了 INT8 的 INT32 输出问题
- E4M3: 适合权重和激活 (范围 ±448)
- E5M2: 适合梯度 (范围更大)
"""

import torch
import torch.nn as nn
import time
import sys


def check_fp8_support():
    """检查 FP8 支持."""
    print("=" * 70)
    print("FP8 支持检查")
    print("=" * 70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu}")
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    
    # 检查架构
    if cap < (8, 9):
        print(f"❌ FP8 需要 Ada Lovelace (sm_89) 或更高")
        print(f"   当前: sm_{cap[0]}{cap[1]}")
        return False
    
    print("✓ GPU 支持 FP8 (Ada Lovelace+)")
    
    # 检查 PyTorch FP8 支持
    has_e4m3 = hasattr(torch, 'float8_e4m3fn')
    has_e5m2 = hasattr(torch, 'float8_e5m2')
    has_scaled_mm = hasattr(torch, '_scaled_mm')
    
    print(f"\ntorch.float8_e4m3fn: {'✓' if has_e4m3 else '❌'}")
    print(f"torch.float8_e5m2:   {'✓' if has_e5m2 else '❌'}")
    print(f"torch._scaled_mm:    {'✓' if has_scaled_mm else '❌'}")
    
    if not (has_e4m3 and has_scaled_mm):
        print("\n❌ 需要 PyTorch >= 2.1 以支持 FP8")
        print("   升级: pip install torch>=2.1")
        return False
    
    print("\n✓ FP8 完全支持!")
    return True


def benchmark(func, num_warmup=20, num_runs=100):
    """Benchmark with CUDA events."""
    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_runs):
        func()
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs


def quantize_to_fp8(x, dtype=torch.float8_e4m3fn):
    """量化到 FP8.
    
    FP8 E4M3 范围: ±448
    FP8 E5M2 范围: ±57344
    """
    if dtype == torch.float8_e4m3fn:
        max_val = 448.0
    else:  # E5M2
        max_val = 57344.0
    
    # 计算 scale 使得 x / scale 在 FP8 范围内
    amax = x.abs().amax()
    scale = amax / max_val
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    
    # 量化
    x_scaled = x / scale
    x_fp8 = x_scaled.to(dtype)
    
    return x_fp8, scale


def quantize_to_fp8_per_token(x, dtype=torch.float8_e4m3fn):
    """Per-token 量化到 FP8."""
    if dtype == torch.float8_e4m3fn:
        max_val = 448.0
    else:
        max_val = 57344.0
    
    # Per-token scale
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / max_val
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    
    x_scaled = x / scale
    x_fp8 = x_scaled.to(dtype)
    
    return x_fp8, scale


def run_basic_comparison():
    """基本对比: FP16 vs FP8."""
    print("\n" + "=" * 70)
    print("1. 基本对比: FP16 vs FP8")
    print("=" * 70)
    
    device = 'cuda'
    M, K, N = 262144, 4096, 4096  # batch=64, seq=4096
    
    print(f"\n矩阵大小: ({M}, {K}) @ ({K}, {N})")
    print(f"TFLOPS: {2 * M * K * N / 1e12:.2f}")
    
    # 创建矩阵
    A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    # FP8 量化
    A_fp8, A_scale = quantize_to_fp8(A_fp16)
    B_fp8, B_scale = quantize_to_fp8(B_fp16)
    
    # Scale tensors
    A_scale_inv = (1.0 / A_scale).to(torch.float32)
    B_scale_inv = (1.0 / B_scale).to(torch.float32)
    
    print(f"\nA_fp8 dtype: {A_fp8.dtype}")
    print(f"B_fp8 dtype: {B_fp8.dtype}")
    print(f"A_scale: {A_scale.item():.6f}")
    print(f"B_scale: {B_scale.item():.6f}")
    
    # FP16 matmul
    fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 50)
    
    # FP8 matmul
    def fp8_matmul():
        return torch._scaled_mm(
            A_fp8, B_fp8,
            scale_a=A_scale_inv,
            scale_b=B_scale_inv,
            out_dtype=torch.float16
        )
    
    fp8_time = benchmark(fp8_matmul, 10, 50)
    
    print(f"\n{'方法':<20} {'时间 (ms)':<15} {'加速比':<10}")
    print("-" * 50)
    print(f"{'FP16 matmul':<20} {fp16_time:<15.4f} {'1.00x':<10}")
    print(f"{'FP8 _scaled_mm':<20} {fp8_time:<15.4f} {fp16_time/fp8_time:.2f}x")
    
    # TFLOPS
    flops = 2 * M * K * N
    fp16_tflops = flops / (fp16_time / 1000) / 1e12
    fp8_tflops = flops / (fp8_time / 1000) / 1e12
    print(f"\nFP16 TFLOPS: {fp16_tflops:.2f}")
    print(f"FP8 TFLOPS:  {fp8_tflops:.2f}")
    
    del A_fp16, B_fp16, A_fp8, B_fp8
    torch.cuda.empty_cache()


def run_quantization_timing():
    """FP8 量化耗时."""
    print("\n" + "=" * 70)
    print("2. FP8 量化/反量化耗时")
    print("=" * 70)
    
    device = 'cuda'
    
    configs = [
        (4096, 4096, "4K x 4K"),
        (16384, 4096, "16K x 4K"),
        (65536, 4096, "64K x 4K"),
        (262144, 4096, "262K x 4K (B64xS4096)"),
        (262144, 1024, "262K x 1K"),
        (262144, 256, "262K x 256 (V latent)"),
    ]
    
    print(f"\n{'配置':<25} {'Per-Tensor Q':<12} {'Per-Token Q':<12} {'Dequant':<12}")
    print("-" * 65)
    
    for M, K, desc in configs:
        try:
            x = torch.randn(M, K, dtype=torch.float16, device=device)
            
            # Per-tensor quantization
            def quant_pt():
                return quantize_to_fp8(x)
            
            pt_time = benchmark(quant_pt, 10, 50)
            
            # Per-token quantization
            def quant_tk():
                return quantize_to_fp8_per_token(x)
            
            tk_time = benchmark(quant_tk, 10, 50)
            
            # Dequantization
            x_fp8, scale = quantize_to_fp8(x)
            
            def dequant():
                return x_fp8.to(torch.float16) * scale
            
            dq_time = benchmark(dequant, 10, 50)
            
            print(f"{desc:<25} {pt_time:<12.4f} {tk_time:<12.4f} {dq_time:<12.4f}")
            
            del x, x_fp8
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")


def run_matrix_size_comparison():
    """不同矩阵大小的对比."""
    print("\n" + "=" * 70)
    print("3. 不同矩阵大小: FP16 vs FP8")
    print("=" * 70)
    
    device = 'cuda'
    
    # (M, K, N, description)
    configs = [
        # 小矩阵
        (1024, 1024, 1024, "1K x 1K x 1K"),
        (4096, 1024, 1024, "4K x 1K x 1K"),
        (4096, 4096, 4096, "4K x 4K x 4K"),
        
        # 中等矩阵
        (16384, 4096, 4096, "16K x 4K x 4K"),
        (65536, 4096, 4096, "64K x 4K x 4K"),
        
        # 大矩阵 (batch=64, seq=4096)
        (262144, 4096, 4096, "262K x 4K x 4K"),
        
        # Value 重建场景 (rank -> hidden)
        (4096, 256, 1024, "4K x 256 -> 1K"),
        (16384, 256, 1024, "16K x 256 -> 1K"),
        (65536, 256, 1024, "64K x 256 -> 1K"),
        (262144, 256, 1024, "262K x 256 -> 1K"),
        
        # 大 hidden dim
        (65536, 256, 4096, "64K x 256 -> 4K"),
        (262144, 256, 4096, "262K x 256 -> 4K"),
        (262144, 512, 4096, "262K x 512 -> 4K"),
    ]
    
    print(f"\n{'配置':<25} {'FP16 (ms)':<12} {'FP8 (ms)':<12} {'FP8 Full':<12} {'Speedup':<10}")
    print("-" * 75)
    
    for M, K, N, desc in configs:
        try:
            A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
            B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
            
            # 预量化
            A_fp8, A_scale = quantize_to_fp8(A_fp16)
            B_fp8, B_scale = quantize_to_fp8(B_fp16)
            A_scale_inv = (1.0 / A_scale).to(torch.float32)
            B_scale_inv = (1.0 / B_scale).to(torch.float32)
            
            # FP16
            fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 50)
            
            # FP8 (预量化)
            def fp8_matmul():
                return torch._scaled_mm(A_fp8, B_fp8, scale_a=A_scale_inv, scale_b=B_scale_inv, out_dtype=torch.float16)
            
            fp8_time = benchmark(fp8_matmul, 10, 50)
            
            # FP8 完整流程 (含量化)
            def fp8_full():
                a_fp8, a_s = quantize_to_fp8(A_fp16)
                a_s_inv = (1.0 / a_s).to(torch.float32)
                return torch._scaled_mm(a_fp8, B_fp8, scale_a=a_s_inv, scale_b=B_scale_inv, out_dtype=torch.float16)
            
            fp8_full_time = benchmark(fp8_full, 10, 50)
            
            speedup = fp16_time / fp8_time
            
            print(f"{desc:<25} {fp16_time:<12.4f} {fp8_time:<12.4f} {fp8_full_time:<12.4f} {speedup:<10.2f}x")
            
            del A_fp16, B_fp16, A_fp8, B_fp8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{desc:<25} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")


def run_batch_seq_comparison():
    """不同 batch/seq 组合的对比."""
    print("\n" + "=" * 70)
    print("4. 不同 Batch/Seq 组合 (K=4096, N=4096)")
    print("=" * 70)
    
    device = 'cuda'
    K, N = 4096, 4096
    
    # 预量化权重
    B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    B_fp8, B_scale = quantize_to_fp8(B_fp16)
    B_scale_inv = (1.0 / B_scale).to(torch.float32)
    
    print(f"\n{'Batch':<8} {'Seq':<8} {'M':<12} {'FP16':<12} {'FP8':<12} {'Speedup':<10}")
    print("-" * 65)
    
    configs = [
        (1, 4096),
        (4, 4096),
        (8, 4096),
        (16, 4096),
        (32, 4096),
        (64, 4096),
        (128, 4096),
        # 短序列大 batch
        (64, 128),
        (64, 256),
        (64, 512),
        (64, 1024),
        (64, 2048),
        (128, 128),
        (128, 256),
        (128, 512),
        (256, 128),
        (256, 256),
    ]
    
    for batch, seq in configs:
        try:
            M = batch * seq
            
            A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
            A_fp8, A_scale = quantize_to_fp8(A_fp16)
            A_scale_inv = (1.0 / A_scale).to(torch.float32)
            
            fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 50)
            
            def fp8_mm():
                return torch._scaled_mm(A_fp8, B_fp8, scale_a=A_scale_inv, scale_b=B_scale_inv, out_dtype=torch.float16)
            
            fp8_time = benchmark(fp8_mm, 10, 50)
            
            speedup = fp16_time / fp8_time
            
            print(f"{batch:<8} {seq:<8} {M:<12} {fp16_time:<12.4f} {fp8_time:<12.4f} {speedup:<10.2f}x")
            
            del A_fp16, A_fp8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:<8} {seq:<8} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{batch:<8} {seq:<8} ERROR: {e}")
    
    del B_fp16, B_fp8
    torch.cuda.empty_cache()


def run_value_reconstruction_comparison():
    """Value 重建场景的对比."""
    print("\n" + "=" * 70)
    print("5. Value 重建场景 (rank -> hidden_dim)")
    print("=" * 70)
    
    device = 'cuda'
    
    print(f"\n{'Batch':<8} {'Seq':<8} {'Rank':<8} {'Hidden':<8} {'FP16':<10} {'FP8':<10} {'Speedup':<10}")
    print("-" * 75)
    
    configs = [
        # 不同 rank
        (64, 4096, 128, 1024),
        (64, 4096, 256, 1024),
        (64, 4096, 512, 1024),
        (64, 4096, 256, 4096),
        (64, 4096, 512, 4096),
        # 不同 batch/seq
        (32, 4096, 256, 1024),
        (128, 4096, 256, 1024),
        (64, 2048, 256, 1024),
        (64, 8192, 256, 1024),
        # 大 batch 短 seq
        (128, 256, 256, 1024),
        (256, 128, 256, 1024),
        (128, 512, 256, 1024),
        (256, 256, 256, 1024),
    ]
    
    for batch, seq, rank, hidden in configs:
        try:
            M = batch * seq
            K = rank
            N = hidden
            
            A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
            B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
            
            A_fp8, A_scale = quantize_to_fp8(A_fp16)
            B_fp8, B_scale = quantize_to_fp8(B_fp16)
            A_scale_inv = (1.0 / A_scale).to(torch.float32)
            B_scale_inv = (1.0 / B_scale).to(torch.float32)
            
            fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 50)
            
            def fp8_mm():
                return torch._scaled_mm(A_fp8, B_fp8, scale_a=A_scale_inv, scale_b=B_scale_inv, out_dtype=torch.float16)
            
            fp8_time = benchmark(fp8_mm, 10, 50)
            
            speedup = fp16_time / fp8_time
            
            print(f"{batch:<8} {seq:<8} {rank:<8} {hidden:<8} {fp16_time:<10.4f} {fp8_time:<10.4f} {speedup:<10.2f}x")
            
            del A_fp16, B_fp16, A_fp8, B_fp8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:<8} {seq:<8} {rank:<8} {hidden:<8} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{batch:<8} {seq:<8} {rank:<8} {hidden:<8} ERROR: {e}")


def run_end_to_end_comparison():
    """端到端对比: FP16 vs FP8 vs INT8."""
    print("\n" + "=" * 70)
    print("6. 端到端对比: FP16 vs FP8 vs INT8")
    print("=" * 70)
    
    device = 'cuda'
    M, K, N = 262144, 4096, 4096
    
    print(f"\n配置: M={M}, K={K}, N={N} (batch=64, seq=4096)")
    
    A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    # 预量化
    A_fp8, A_scale = quantize_to_fp8(A_fp16)
    B_fp8, B_scale = quantize_to_fp8(B_fp16)
    A_scale_inv = (1.0 / A_scale).to(torch.float32)
    B_scale_inv = (1.0 / B_scale).to(torch.float32)
    
    A_int8 = (A_fp16 / (A_fp16.abs().amax() / 127)).round().clamp(-128, 127).to(torch.int8)
    B_int8 = (B_fp16 / (B_fp16.abs().amax() / 127)).round().clamp(-128, 127).to(torch.int8)
    
    # FP16
    fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 50)
    
    # FP8 (预量化)
    def fp8_mm():
        return torch._scaled_mm(A_fp8, B_fp8, scale_a=A_scale_inv, scale_b=B_scale_inv, out_dtype=torch.float16)
    
    fp8_time = benchmark(fp8_mm, 10, 50)
    
    # INT8 (预量化)
    int8_time = benchmark(lambda: torch._int_mm(A_int8, B_int8), 10, 50)
    
    # FP8 完整流程
    def fp8_full():
        a_fp8, a_s = quantize_to_fp8(A_fp16)
        a_s_inv = (1.0 / a_s).to(torch.float32)
        return torch._scaled_mm(a_fp8, B_fp8, scale_a=a_s_inv, scale_b=B_scale_inv, out_dtype=torch.float16)
    
    fp8_full_time = benchmark(fp8_full, 10, 50)
    
    print(f"\n{'方法':<30} {'时间 (ms)':<15} {'加速比':<10}")
    print("-" * 60)
    print(f"{'FP16 matmul':<30} {fp16_time:<15.4f} {'1.00x':<10}")
    print(f"{'FP8 (预量化)':<30} {fp8_time:<15.4f} {fp16_time/fp8_time:.2f}x")
    print(f"{'FP8 (含量化)':<30} {fp8_full_time:<15.4f} {fp16_time/fp8_full_time:.2f}x")
    print(f"{'INT8 (预量化, INT32输出)':<30} {int8_time:<15.4f} {fp16_time/int8_time:.2f}x")
    
    # TFLOPS
    flops = 2 * M * K * N
    print(f"\nTFLOPS/TOPS:")
    print(f"  FP16: {flops / (fp16_time / 1000) / 1e12:.2f} TFLOPS")
    print(f"  FP8:  {flops / (fp8_time / 1000) / 1e12:.2f} TFLOPS")
    print(f"  INT8: {flops / (int8_time / 1000) / 1e12:.2f} TOPS")
    
    del A_fp16, B_fp16, A_fp8, B_fp8, A_int8, B_int8
    torch.cuda.empty_cache()


def run_quantization_error_comparison():
    """量化误差对比: FP8 vs INT8."""
    print("\n" + "=" * 70)
    print("7. 量化误差对比: FP8 vs INT8")
    print("=" * 70)
    
    device = 'cuda'
    
    x = torch.randn(1024, 1024, dtype=torch.float16, device=device)
    
    # FP8 E4M3
    x_fp8_e4m3, scale_e4m3 = quantize_to_fp8(x, torch.float8_e4m3fn)
    x_dequant_e4m3 = x_fp8_e4m3.to(torch.float16) * scale_e4m3
    error_e4m3 = (x - x_dequant_e4m3).abs()
    
    # FP8 E5M2
    x_fp8_e5m2, scale_e5m2 = quantize_to_fp8(x, torch.float8_e5m2)
    x_dequant_e5m2 = x_fp8_e5m2.to(torch.float16) * scale_e5m2
    error_e5m2 = (x - x_dequant_e5m2).abs()
    
    # INT8
    int8_scale = x.abs().amax() / 127
    x_int8 = (x / int8_scale).round().clamp(-128, 127).to(torch.int8)
    x_dequant_int8 = x_int8.half() * int8_scale
    error_int8 = (x - x_dequant_int8).abs()
    
    print(f"\n{'量化方法':<20} {'Mean Error':<15} {'Max Error':<15} {'Rel Error':<15}")
    print("-" * 70)
    print(f"{'FP8 E4M3':<20} {error_e4m3.mean().item():<15.6f} {error_e4m3.max().item():<15.6f} {(error_e4m3 / (x.abs() + 1e-6)).mean().item():<15.4%}")
    print(f"{'FP8 E5M2':<20} {error_e5m2.mean().item():<15.6f} {error_e5m2.max().item():<15.6f} {(error_e5m2 / (x.abs() + 1e-6)).mean().item():<15.4%}")
    print(f"{'INT8':<20} {error_int8.mean().item():<15.6f} {error_int8.max().item():<15.6f} {(error_int8 / (x.abs() + 1e-6)).mean().item():<15.4%}")
    
    print("\n说明:")
    print("  - FP8 E4M3: 范围 ±448, 3位尾数, 适合权重和激活")
    print("  - FP8 E5M2: 范围 ±57344, 2位尾数, 适合梯度")
    print("  - INT8: 范围 ±127, 均匀量化")


def main():
    if not check_fp8_support():
        print("\n无法进行 FP8 测试")
        sys.exit(1)
    
    run_basic_comparison()
    run_quantization_timing()
    run_matrix_size_comparison()
    run_batch_seq_comparison()
    run_value_reconstruction_comparison()
    run_end_to_end_comparison()
    run_quantization_error_comparison()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
FP8 vs INT8 关键区别:

1. **输出数据类型**:
   - FP8: 输出直接是 FP16 (2 bytes)
   - INT8: 输出是 INT32 (4 bytes) -> 需要额外转换
   
2. **内存带宽**:
   - FP8: 读 1 byte, 写 2 bytes
   - INT8: 读 1 byte, 写 4 bytes (然后还要转换)
   
3. **精度**:
   - FP8 E4M3: 浮点量化，对分布不均匀的数据更友好
   - INT8: 均匀量化，可能需要更细粒度的 scale
   
4. **硬件支持**:
   - FP8: Ada Lovelace+ (L20, RTX 40xx) 原生支持
   - INT8: 所有支持 Tensor Core 的 GPU

建议:
- 如果有 Ada Lovelace GPU (L20)，优先使用 FP8
- FP8 E4M3 适合前向传播
- 大矩阵时 FP8 加速更明显
    """)


if __name__ == "__main__":
    main()
