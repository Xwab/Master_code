"""
测试 batch=64, seq_len=4096, proj_dim=4096 的矩阵乘法: bitsandbytes INT8 vs FP16

矩阵大小: (64*4096, 4096) @ (4096, 4096) = (262144, 4096)
"""

import torch
import torch.nn as nn
import time
import sys

def check_bnb():
    """Check bitsandbytes availability."""
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes version: {bnb.__version__}")
        return True
    except ImportError:
        print("bitsandbytes not installed. Run: pip install bitsandbytes")
        return False


def benchmark(func, num_warmup=20, num_runs=100):
    """Benchmark with CUDA events for accurate timing."""
    # Warmup
    for _ in range(num_warmup):
        func()
    torch.cuda.synchronize()
    
    # Use CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_runs):
        func()
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs


def main():
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F
    
    print("=" * 70)
    print("测试: batch=64, seq_len=4096, proj_dim=4096 (bitsandbytes)")
    print("=" * 70)
    
    device = 'cuda'
    
    # 环境信息
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bitsandbytes: {bnb.__version__}")
    
    # 矩阵大小
    batch = 64
    seq_len = 4096
    M = batch * seq_len  # 262144
    K = 4096  # 输入维度
    N = 4096  # 输出维度
    
    print(f"\nbatch_size: {batch}")
    print(f"seq_len: {seq_len}")
    print(f"M (total tokens): {M} = {batch} * {seq_len}")
    print(f"K (input dim): {K}")
    print(f"N (output dim): {N}")
    print(f"\n矩阵大小: A({M}, {K}) @ B({K}, {N}) = C({M}, {N})")
    print(f"TFLOPS: {2 * M * K * N / 1e12:.2f}")
    
    # ================================================================
    # 测试 1: bnb.nn.Linear8bitLt
    # ================================================================
    print("\n" + "=" * 70)
    print("测试 1: bnb.nn.Linear8bitLt vs nn.Linear")
    print("=" * 70)
    
    # 创建 FP16 Linear
    linear_fp16 = nn.Linear(K, N, bias=False).cuda().half()
    
    # 创建 BNB INT8 Linear
    linear_int8 = bnb.nn.Linear8bitLt(
        K, N,
        bias=False,
        has_fp16_weights=False,
        threshold=6.0,
    ).cuda()
    
    # 复制权重
    linear_int8.weight = bnb.nn.Int8Params(
        linear_fp16.weight.data.clone(),
        requires_grad=False,
        has_fp16_weights=False,
    ).cuda()
    
    # 创建输入
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    
    print(f"\n输入 x shape: {x.shape}")
    print(f"FP16 权重 shape: {linear_fp16.weight.shape}")
    
    # Benchmark
    fp16_time = benchmark(lambda: linear_fp16(x), num_warmup=10, num_runs=50)
    int8_time = benchmark(lambda: linear_int8(x), num_warmup=10, num_runs=50)
    
    print(f"\nnn.Linear (FP16):        {fp16_time:.4f} ms")
    print(f"bnb.Linear8bitLt (INT8): {int8_time:.4f} ms")
    print(f"加速比: {fp16_time / int8_time:.2f}x")
    
    # TFLOPS
    flops = 2 * M * K * N
    fp16_tflops = flops / (fp16_time / 1000) / 1e12
    int8_tops = flops / (int8_time / 1000) / 1e12
    print(f"\nFP16 TFLOPS: {fp16_tflops:.2f}")
    print(f"INT8 TOPS:   {int8_tops:.2f}")
    
    del linear_fp16, linear_int8, x
    torch.cuda.empty_cache()
    
    # ================================================================
    # 测试 2: bnb.functional 量化/反量化
    # ================================================================
    print("\n" + "=" * 70)
    print("测试 2: bnb.functional 量化/反量化耗时")
    print("=" * 70)
    
    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(K, N, dtype=torch.float16, device=device)
    
    # 量化 A
    def quant_A():
        return bnb_F.quantize_blockwise(A)
    
    # 量化 B
    def quant_B():
        return bnb_F.quantize_blockwise(B)
    
    quant_A_time = benchmark(quant_A, num_warmup=10, num_runs=50)
    quant_B_time = benchmark(quant_B, num_warmup=10, num_runs=50)
    
    print(f"\n量化 A ({M}x{K}): {quant_A_time:.4f} ms")
    print(f"量化 B ({K}x{N}): {quant_B_time:.4f} ms")
    
    # 预量化
    A_int8, state_A = bnb_F.quantize_blockwise(A)
    B_int8, state_B = bnb_F.quantize_blockwise(B)
    
    # 反量化
    def dequant_A():
        return bnb_F.dequantize_blockwise(A_int8, state_A)
    
    def dequant_B():
        return bnb_F.dequantize_blockwise(B_int8, state_B)
    
    dequant_A_time = benchmark(dequant_A, num_warmup=10, num_runs=50)
    dequant_B_time = benchmark(dequant_B, num_warmup=10, num_runs=50)
    
    print(f"\n反量化 A: {dequant_A_time:.4f} ms")
    print(f"反量化 B: {dequant_B_time:.4f} ms")
    
    del A_int8, B_int8, state_A, state_B
    torch.cuda.empty_cache()
    
    # ================================================================
    # 测试 3: 完整流程对比
    # ================================================================
    print("\n" + "=" * 70)
    print("测试 3: 完整流程对比 (含量化/反量化)")
    print("=" * 70)
    
    # 预量化 B (权重)
    B_int8, state_B = bnb_F.quantize_blockwise(B)
    
    # FP16 baseline
    def fp16_matmul():
        return torch.matmul(A, B)
    
    fp16_time = benchmark(fp16_matmul, num_warmup=10, num_runs=50)
    
    # BNB: 反量化 B + FP16 matmul
    def bnb_dequant_matmul():
        B_dequant = bnb_F.dequantize_blockwise(B_int8, state_B)
        return torch.matmul(A, B_dequant)
    
    bnb_dequant_time = benchmark(bnb_dequant_matmul, num_warmup=10, num_runs=50)
    
    print(f"\nFP16 matmul:                    {fp16_time:.4f} ms")
    print(f"BNB (dequant B + FP16 matmul):  {bnb_dequant_time:.4f} ms")
    print(f"加速比: {fp16_time / bnb_dequant_time:.2f}x")
    
    del B_int8, state_B
    torch.cuda.empty_cache()
    
    # ================================================================
    # 测试 4: 与 torch._int_mm 对比
    # ================================================================
    print("\n" + "=" * 70)
    print("测试 4: bitsandbytes vs torch._int_mm vs FP16")
    print("=" * 70)
    
    # 重新创建 Linear 层
    linear_fp16 = nn.Linear(K, N, bias=False).cuda().half()
    
    linear_bnb = bnb.nn.Linear8bitLt(K, N, bias=False, has_fp16_weights=False).cuda()
    linear_bnb.weight = bnb.nn.Int8Params(
        linear_fp16.weight.data.clone(),
        requires_grad=False,
        has_fp16_weights=False,
    ).cuda()
    
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    W = linear_fp16.weight.data  # (N, K)
    
    # 预量化 for torch._int_mm
    W_t = W.t().contiguous()  # (K, N) for matmul
    W_scale = W_t.abs().amax() / 127 + 1e-6
    W_int8 = (W_t / W_scale).round().clamp(-128, 127).to(torch.int8)
    
    x_scale = x.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
    x_int8 = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
    
    # FP16
    fp16_time = benchmark(lambda: linear_fp16(x), num_warmup=10, num_runs=50)
    
    # BNB
    bnb_time = benchmark(lambda: linear_bnb(x), num_warmup=10, num_runs=50)
    
    # torch._int_mm (预量化)
    int_mm_time = benchmark(lambda: torch._int_mm(x_int8, W_int8), num_warmup=10, num_runs=50)
    
    # torch._int_mm (完整流程)
    def torch_int8_full():
        xs = x.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
        xi = (x / xs).round().clamp(-128, 127).to(torch.int8)
        c = torch._int_mm(xi, W_int8)
        return c.half() * xs * W_scale
    
    torch_int8_full_time = benchmark(torch_int8_full, num_warmup=10, num_runs=50)
    
    print(f"\n{'方法':<35} {'时间 (ms)':<15} {'加速比':<10}")
    print("-" * 65)
    print(f"{'FP16 nn.Linear':<35} {fp16_time:<15.4f} {'1.00x':<10}")
    print(f"{'bitsandbytes Linear8bitLt':<35} {bnb_time:<15.4f} {fp16_time/bnb_time:.2f}x")
    print(f"{'torch._int_mm (预量化)':<35} {int_mm_time:<15.4f} {fp16_time/int_mm_time:.2f}x")
    print(f"{'torch._int_mm (完整流程)':<35} {torch_int8_full_time:<15.4f} {fp16_time/torch_int8_full_time:.2f}x")
    
    del linear_fp16, linear_bnb, x, W, W_t, W_int8, x_int8
    torch.cuda.empty_cache()
    
    # ================================================================
    # 测试 5: 不同 batch size
    # ================================================================
    print("\n" + "=" * 70)
    print("测试 5: 不同 batch_size (seq_len=4096, K=4096, N=4096)")
    print("=" * 70)
    
    print(f"\n{'Batch':<10} {'M':<12} {'FP16':<12} {'BNB':<12} {'_int_mm':<12} {'BNB Spd':<10} {'int_mm Spd':<10}")
    print("-" * 80)
    
    # 重新创建权重
    W_fp16 = torch.randn(N, K, dtype=torch.float16, device=device)
    W_t = W_fp16.t().contiguous()
    W_scale = W_t.abs().amax() / 127 + 1e-6
    W_int8 = (W_t / W_scale).round().clamp(-128, 127).to(torch.int8)
    
    test_batches = [1, 4, 8, 16, 32, 64, 128]
    
    for test_batch in test_batches:
        try:
            test_M = test_batch * seq_len
            
            # 创建输入
            x = torch.randn(test_M, K, dtype=torch.float16, device=device)
            
            # FP16 Linear
            linear_fp16 = nn.Linear(K, N, bias=False).cuda().half()
            linear_fp16.weight.data = W_fp16.clone()
            
            # BNB Linear
            linear_bnb = bnb.nn.Linear8bitLt(K, N, bias=False, has_fp16_weights=False).cuda()
            linear_bnb.weight = bnb.nn.Int8Params(
                W_fp16.clone(), requires_grad=False, has_fp16_weights=False
            ).cuda()
            
            # 预量化 x
            x_scale = x.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
            x_int8 = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
            
            # Benchmark
            t_fp16 = benchmark(lambda: linear_fp16(x), num_warmup=5, num_runs=20)
            t_bnb = benchmark(lambda: linear_bnb(x), num_warmup=5, num_runs=20)
            t_int_mm = benchmark(lambda: torch._int_mm(x_int8, W_int8), num_warmup=5, num_runs=20)
            
            bnb_spd = t_fp16 / t_bnb
            int_mm_spd = t_fp16 / t_int_mm
            
            print(f"{test_batch:<10} {test_M:<12} {t_fp16:<12.4f} {t_bnb:<12.4f} {t_int_mm:<12.4f} {bnb_spd:<10.2f}x {int_mm_spd:<10.2f}x")
            
            del linear_fp16, linear_bnb, x, x_int8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{test_batch:<10} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{test_batch:<10} ERROR: {e}")
    
    # ================================================================
    # 测试 6: 不同 seq_len
    # ================================================================
    print("\n" + "=" * 70)
    print("测试 6: 不同 seq_len (batch=64, K=4096, N=4096)")
    print("=" * 70)
    
    print(f"\n{'Seq Len':<10} {'M':<12} {'FP16':<12} {'BNB':<12} {'_int_mm':<12} {'BNB Spd':<10} {'int_mm Spd':<10}")
    print("-" * 80)
    
    test_seq_lens = [128, 256, 512, 1024, 2048, 4096]
    
    for test_seq in test_seq_lens:
        try:
            test_M = batch * test_seq
            
            x = torch.randn(test_M, K, dtype=torch.float16, device=device)
            
            linear_fp16 = nn.Linear(K, N, bias=False).cuda().half()
            linear_fp16.weight.data = W_fp16.clone()
            
            linear_bnb = bnb.nn.Linear8bitLt(K, N, bias=False, has_fp16_weights=False).cuda()
            linear_bnb.weight = bnb.nn.Int8Params(
                W_fp16.clone(), requires_grad=False, has_fp16_weights=False
            ).cuda()
            
            x_scale = x.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
            x_int8 = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
            
            t_fp16 = benchmark(lambda: linear_fp16(x), num_warmup=5, num_runs=20)
            t_bnb = benchmark(lambda: linear_bnb(x), num_warmup=5, num_runs=20)
            t_int_mm = benchmark(lambda: torch._int_mm(x_int8, W_int8), num_warmup=5, num_runs=20)
            
            bnb_spd = t_fp16 / t_bnb
            int_mm_spd = t_fp16 / t_int_mm
            
            print(f"{test_seq:<10} {test_M:<12} {t_fp16:<12.4f} {t_bnb:<12.4f} {t_int_mm:<12.4f} {bnb_spd:<10.2f}x {int_mm_spd:<10.2f}x")
            
            del linear_fp16, linear_bnb, x, x_int8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{test_seq:<10} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{test_seq:<10} ERROR: {e}")
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"""
配置: batch={batch}, seq_len={seq_len}, K={K}, N={N}
矩阵: ({M}, {K}) @ ({K}, {N})

对比:
1. bitsandbytes Linear8bitLt: 封装好的 INT8 Linear，内部使用优化的 CUDA kernel
2. torch._int_mm: PyTorch 原生 INT8 GEMM，使用 cuBLASLt

预期:
- 大矩阵 (M >= 64K) 时，INT8 应该有明显加速
- bitsandbytes 和 torch._int_mm 性能应该接近
- 量化/反量化开销在大矩阵时占比小

注意:
- bitsandbytes 使用 blockwise 量化，精度可能更好
- torch._int_mm 使用简单的 per-tensor/per-token 量化
""")


if __name__ == "__main__":
    if not check_bnb():
        sys.exit(1)
    
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        main()
    else:
        print("CUDA not available!")
