"""
测试 batch=64, seq_len=4096, proj_dim=4096 的矩阵乘法: INT8 vs FP16

矩阵大小: (64*4096, 4096) @ (4096, 4096) = (262144, 4096)
"""

import torch
import time

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
    print("=" * 70)
    print("测试: batch=64, seq_len=4096, proj_dim=4096")
    print("=" * 70)
    
    device = 'cuda'
    
    # 环境信息
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    
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
    print(f"GFLOPS: {2 * M * K * N / 1e9:.2f}")
    print(f"A 内存: {M * K * 2 / 1e9:.2f} GB (FP16)")
    print(f"B 内存: {K * N * 2 / 1e6:.2f} MB (FP16)")
    print(f"C 内存: {M * N * 2 / 1e9:.2f} GB (FP16)")
    
    # 创建矩阵
    print("\n创建矩阵...")
    A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    print(f"A shape: {A_fp16.shape}, dtype: {A_fp16.dtype}")
    print(f"B shape: {B_fp16.shape}, dtype: {B_fp16.dtype}")
    
    # 量化
    print("\n" + "-" * 70)
    print("1. 量化耗时")
    print("-" * 70)
    
    # Per-tensor 量化 A
    def quant_A_per_tensor():
        scale = A_fp16.abs().amax() / 127 + 1e-6
        return (A_fp16 / scale).round().clamp(-128, 127).to(torch.int8), scale
    
    # Per-token 量化 A (KIVI-style for Value)
    def quant_A_per_token():
        scale = A_fp16.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
        return (A_fp16 / scale).round().clamp(-128, 127).to(torch.int8), scale
    
    # 量化 B (权重，可以离线预计算)
    def quant_B():
        scale = B_fp16.abs().amax() / 127 + 1e-6
        return (B_fp16 / scale).round().clamp(-128, 127).to(torch.int8), scale
    
    quant_A_pt_time = benchmark(quant_A_per_tensor, num_warmup=10, num_runs=50)
    quant_A_tk_time = benchmark(quant_A_per_token, num_warmup=10, num_runs=50)
    quant_B_time = benchmark(quant_B, num_warmup=10, num_runs=50)
    
    print(f"量化 A (per-tensor): {quant_A_pt_time:.4f} ms")
    print(f"量化 A (per-token):  {quant_A_tk_time:.4f} ms  (KIVI Value style)")
    print(f"量化 B (per-tensor): {quant_B_time:.4f} ms  (可离线预计算)")
    
    # 预量化
    print("\n预量化中...")
    A_scale_pt = A_fp16.abs().amax() / 127 + 1e-6
    A_int8_pt = (A_fp16 / A_scale_pt).round().clamp(-128, 127).to(torch.int8)
    
    A_scale_tk = A_fp16.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
    A_int8_tk = (A_fp16 / A_scale_tk).round().clamp(-128, 127).to(torch.int8)
    
    B_scale = B_fp16.abs().amax() / 127 + 1e-6
    B_int8 = (B_fp16 / B_scale).round().clamp(-128, 127).to(torch.int8)
    
    print(f"A_int8 shape: {A_int8_pt.shape}, dtype: {A_int8_pt.dtype}")
    print(f"B_int8 shape: {B_int8.shape}, dtype: {B_int8.dtype}")
    
    print("\n" + "-" * 70)
    print("2. 纯矩阵乘法耗时 (预量化，无量化开销)")
    print("-" * 70)
    
    # FP16 matmul
    fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), num_warmup=10, num_runs=50)
    
    # INT8 matmul
    int8_time = benchmark(lambda: torch._int_mm(A_int8_pt, B_int8), num_warmup=10, num_runs=50)
    
    print(f"FP16 torch.matmul:  {fp16_time:.4f} ms")
    print(f"INT8 torch._int_mm: {int8_time:.4f} ms")
    print(f"纯计算加速比: {fp16_time / int8_time:.2f}x")
    
    # TFLOPS/TOPS
    flops = 2 * M * K * N
    fp16_tflops = flops / (fp16_time / 1000) / 1e12
    int8_tops = flops / (int8_time / 1000) / 1e12
    print(f"\nFP16 TFLOPS: {fp16_tflops:.2f}")
    print(f"INT8 TOPS:   {int8_tops:.2f}")
    print(f"理论 INT8/FP16 吞吐比: {int8_tops / fp16_tflops:.2f}x")
    
    print("\n" + "-" * 70)
    print("3. 反量化耗时")
    print("-" * 70)
    
    C_int32_pt = torch._int_mm(A_int8_pt, B_int8)
    C_int32_tk = torch._int_mm(A_int8_tk, B_int8)
    
    print(f"C_int32 shape: {C_int32_pt.shape}, dtype: {C_int32_pt.dtype}")
    
    # 反量化 (per-tensor)
    def dequant_pt():
        return C_int32_pt.half() * A_scale_pt * B_scale
    
    # 反量化 (per-token)
    def dequant_tk():
        return C_int32_tk.half() * A_scale_tk * B_scale
    
    dequant_pt_time = benchmark(dequant_pt, num_warmup=10, num_runs=50)
    dequant_tk_time = benchmark(dequant_tk, num_warmup=10, num_runs=50)
    
    print(f"反量化 (per-tensor): {dequant_pt_time:.4f} ms")
    print(f"反量化 (per-token):  {dequant_tk_time:.4f} ms")
    
    print("\n" + "-" * 70)
    print("4. 端到端耗时对比 (预量化权重 B)")
    print("-" * 70)
    
    print(f"\n{'方法':<25} {'时间 (ms)':<15} {'加速比':<10}")
    print("-" * 55)
    
    # FP16 baseline
    print(f"{'FP16 (baseline)':<25} {fp16_time:<15.4f} {'1.00x':<10}")
    
    # INT8 per-tensor
    total_int8_pt = quant_A_pt_time + int8_time + dequant_pt_time
    speedup_pt = fp16_time / total_int8_pt
    print(f"{'INT8 Per-Tensor':<25} {total_int8_pt:<15.4f} {speedup_pt:.2f}x")
    
    # INT8 per-token (KIVI-style)
    total_int8_tk = quant_A_tk_time + int8_time + dequant_tk_time
    speedup_tk = fp16_time / total_int8_tk
    print(f"{'INT8 Per-Token (KIVI)':<25} {total_int8_tk:<15.4f} {speedup_tk:.2f}x")
    
    print("\n详细分解 (INT8 Per-Token):")
    print(f"  量化 A (per-token): {quant_A_tk_time:.4f} ms ({quant_A_tk_time/total_int8_tk*100:.1f}%)")
    print(f"  INT8 _int_mm:       {int8_time:.4f} ms ({int8_time/total_int8_tk*100:.1f}%)")
    print(f"  反量化:             {dequant_tk_time:.4f} ms ({dequant_tk_time/total_int8_tk*100:.1f}%)")
    print(f"  总计:               {total_int8_tk:.4f} ms")
    
    print("\n" + "-" * 70)
    print("5. 完整流程验证 (实际调用)")
    print("-" * 70)
    
    def full_int8_per_tensor():
        a_s = A_fp16.abs().amax() / 127 + 1e-6
        a_i = (A_fp16 / a_s).round().clamp(-128, 127).to(torch.int8)
        c = torch._int_mm(a_i, B_int8)
        return c.half() * a_s * B_scale
    
    def full_int8_per_token():
        a_s = A_fp16.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
        a_i = (A_fp16 / a_s).round().clamp(-128, 127).to(torch.int8)
        c = torch._int_mm(a_i, B_int8)
        return c.half() * a_s * B_scale
    
    full_pt_time = benchmark(full_int8_per_tensor, num_warmup=10, num_runs=50)
    full_tk_time = benchmark(full_int8_per_token, num_warmup=10, num_runs=50)
    
    print(f"{'FP16':<25} {fp16_time:<15.4f} {'1.00x (baseline)':<20}")
    print(f"{'INT8 Per-Tensor (full)':<25} {full_pt_time:<15.4f} {fp16_time/full_pt_time:.2f}x")
    print(f"{'INT8 Per-Token (full)':<25} {full_tk_time:<15.4f} {fp16_time/full_tk_time:.2f}x")
    
    # 清理中间变量
    del C_int32_pt, C_int32_tk
    torch.cuda.empty_cache()
    
    print("\n" + "-" * 70)
    print("6. 不同 batch_size 测试 (seq_len=4096, K=4096, N=4096)")
    print("-" * 70)
    
    print(f"\n{'Batch':<10} {'Total M':<12} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'Full INT8':<12} {'Speedup':<10}")
    print("-" * 70)
    
    test_batches = [1, 4, 8, 16, 32, 64, 128]
    
    for test_batch in test_batches:
        try:
            test_M = test_batch * seq_len
            
            A_t = torch.randn(test_M, K, dtype=torch.float16, device=device)
            
            # 预量化
            A_s = A_t.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
            A_i = (A_t / A_s).round().clamp(-128, 127).to(torch.int8)
            
            # FP16
            t_fp16 = benchmark(lambda: torch.matmul(A_t, B_fp16), num_warmup=5, num_runs=20)
            
            # INT8 only
            t_int8 = benchmark(lambda: torch._int_mm(A_i, B_int8), num_warmup=5, num_runs=20)
            
            # Full INT8
            def full_int8():
                a_s = A_t.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
                a_i = (A_t / a_s).round().clamp(-128, 127).to(torch.int8)
                c = torch._int_mm(a_i, B_int8)
                return c.half() * a_s * B_scale
            
            t_full = benchmark(full_int8, num_warmup=5, num_runs=20)
            
            speedup = t_fp16 / t_int8
            
            print(f"{test_batch:<10} {test_M:<12} {t_fp16:<12.4f} {t_int8:<12.4f} {t_full:<12.4f} {speedup:<10.2f}x")
            
            del A_t, A_i
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{test_batch:<10} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{test_batch:<10} ERROR: {e}")
    
    print("\n" + "-" * 70)
    print("7. 不同 seq_len 测试 (batch=64, K=4096, N=4096)")
    print("-" * 70)
    
    print(f"\n{'Seq Len':<10} {'Total M':<12} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'Full INT8':<12} {'Speedup':<10}")
    print("-" * 70)
    
    test_seq_lens = [256, 512, 1024, 2048, 4096]
    
    for test_seq in test_seq_lens:
        try:
            test_M = batch * test_seq
            
            A_t = torch.randn(test_M, K, dtype=torch.float16, device=device)
            
            A_s = A_t.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
            A_i = (A_t / A_s).round().clamp(-128, 127).to(torch.int8)
            
            t_fp16 = benchmark(lambda: torch.matmul(A_t, B_fp16), num_warmup=5, num_runs=20)
            t_int8 = benchmark(lambda: torch._int_mm(A_i, B_int8), num_warmup=5, num_runs=20)
            
            def full_int8():
                a_s = A_t.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
                a_i = (A_t / a_s).round().clamp(-128, 127).to(torch.int8)
                c = torch._int_mm(a_i, B_int8)
                return c.half() * a_s * B_scale
            
            t_full = benchmark(full_int8, num_warmup=5, num_runs=20)
            
            speedup = t_fp16 / t_int8
            
            print(f"{test_seq:<10} {test_M:<12} {t_fp16:<12.4f} {t_int8:<12.4f} {t_full:<12.4f} {speedup:<10.2f}x")
            
            del A_t, A_i
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
总计算量: {2 * M * K * N / 1e12:.2f} TFLOPS

结果:
- FP16 matmul:           {fp16_time:.4f} ms
- INT8 _int_mm (纯计算):  {int8_time:.4f} ms  (加速 {fp16_time/int8_time:.2f}x)
- INT8 端到端 (per-token): {full_tk_time:.4f} ms  (加速 {fp16_time/full_tk_time:.2f}x)

开销分析:
- 量化 (per-token):  {quant_A_tk_time:.4f} ms ({quant_A_tk_time/full_tk_time*100:.1f}%)
- INT8 计算:         {int8_time:.4f} ms ({int8_time/full_tk_time*100:.1f}%)
- 反量化:            {dequant_tk_time:.4f} ms ({dequant_tk_time/full_tk_time*100:.1f}%)
""")


if __name__ == "__main__":
    main()
