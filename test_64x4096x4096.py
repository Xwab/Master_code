"""
测试 64 * 4096 * 4096 矩阵乘法: INT8 vs FP16
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
    print("测试: 64 x 4096 @ 4096 x 4096 矩阵乘法")
    print("=" * 70)
    
    device = 'cuda'
    
    # 环境信息
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # 矩阵大小
    M = 64 * 64  # batch * seq_len = 4096
    K = 4096
    N = 4096
    
    print(f"\n矩阵大小: A({M}, {K}) @ B({K}, {N}) = C({M}, {N})")
    print(f"总 tokens: {M} (例如 batch=64, seq_len=64)")
    print(f"FLOPS: {2 * M * K * N / 1e9:.2f} GFLOPS")
    
    # 创建矩阵
    A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    # 量化
    print("\n" + "-" * 70)
    print("1. 量化耗时")
    print("-" * 70)
    
    # Per-tensor 量化 A
    def quant_A_per_tensor():
        scale = A_fp16.abs().amax() / 127 + 1e-6
        return (A_fp16 / scale).round().clamp(-128, 127).to(torch.int8), scale
    
    # Per-token 量化 A
    def quant_A_per_token():
        scale = A_fp16.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
        return (A_fp16 / scale).round().clamp(-128, 127).to(torch.int8), scale
    
    # 量化 B
    def quant_B():
        scale = B_fp16.abs().amax() / 127 + 1e-6
        return (B_fp16 / scale).round().clamp(-128, 127).to(torch.int8), scale
    
    quant_A_pt_time = benchmark(quant_A_per_tensor)
    quant_A_tk_time = benchmark(quant_A_per_token)
    quant_B_time = benchmark(quant_B)
    
    print(f"量化 A (per-tensor): {quant_A_pt_time:.4f} ms")
    print(f"量化 A (per-token):  {quant_A_tk_time:.4f} ms")
    print(f"量化 B (per-tensor): {quant_B_time:.4f} ms")
    
    # 预量化
    A_scale_pt = A_fp16.abs().amax() / 127 + 1e-6
    A_int8_pt = (A_fp16 / A_scale_pt).round().clamp(-128, 127).to(torch.int8)
    
    A_scale_tk = A_fp16.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
    A_int8_tk = (A_fp16 / A_scale_tk).round().clamp(-128, 127).to(torch.int8)
    
    B_scale = B_fp16.abs().amax() / 127 + 1e-6
    B_int8 = (B_fp16 / B_scale).round().clamp(-128, 127).to(torch.int8)
    
    print("\n" + "-" * 70)
    print("2. 纯矩阵乘法耗时 (预量化，无量化开销)")
    print("-" * 70)
    
    # FP16 matmul
    fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16))
    
    # INT8 matmul
    int8_time = benchmark(lambda: torch._int_mm(A_int8_pt, B_int8))
    
    print(f"FP16 matmul:  {fp16_time:.4f} ms")
    print(f"INT8 _int_mm: {int8_time:.4f} ms")
    print(f"加速比 (INT8 vs FP16): {fp16_time / int8_time:.2f}x")
    
    # TFLOPS
    flops = 2 * M * K * N
    fp16_tflops = flops / (fp16_time / 1000) / 1e12
    int8_tops = flops / (int8_time / 1000) / 1e12
    print(f"\nFP16 TFLOPS: {fp16_tflops:.2f}")
    print(f"INT8 TOPS:   {int8_tops:.2f}")
    
    print("\n" + "-" * 70)
    print("3. 反量化耗时")
    print("-" * 70)
    
    C_int32 = torch._int_mm(A_int8_pt, B_int8)
    
    # 反量化 (per-tensor)
    def dequant_pt():
        return C_int32.half() * A_scale_pt * B_scale
    
    # 反量化 (per-token)
    C_int32_tk = torch._int_mm(A_int8_tk, B_int8)
    def dequant_tk():
        return C_int32_tk.half() * A_scale_tk * B_scale
    
    dequant_pt_time = benchmark(dequant_pt)
    dequant_tk_time = benchmark(dequant_tk)
    
    print(f"反量化 (per-tensor): {dequant_pt_time:.4f} ms")
    print(f"反量化 (per-token):  {dequant_tk_time:.4f} ms")
    
    print("\n" + "-" * 70)
    print("4. 端到端耗时对比")
    print("-" * 70)
    
    # FP16 baseline
    print(f"\nFP16 (baseline): {fp16_time:.4f} ms")
    
    # INT8 per-tensor (预量化 B)
    total_int8_pt = quant_A_pt_time + int8_time + dequant_pt_time
    print(f"\nINT8 Per-Tensor (预量化权重):")
    print(f"  量化 A:   {quant_A_pt_time:.4f} ms")
    print(f"  _int_mm:  {int8_time:.4f} ms")
    print(f"  反量化:   {dequant_pt_time:.4f} ms")
    print(f"  总计:     {total_int8_pt:.4f} ms")
    print(f"  加速比:   {fp16_time / total_int8_pt:.2f}x")
    
    # INT8 per-token (预量化 B)
    total_int8_tk = quant_A_tk_time + int8_time + dequant_tk_time
    print(f"\nINT8 Per-Token (预量化权重, KIVI-style):")
    print(f"  量化 A:   {quant_A_tk_time:.4f} ms")
    print(f"  _int_mm:  {int8_time:.4f} ms")
    print(f"  反量化:   {dequant_tk_time:.4f} ms")
    print(f"  总计:     {total_int8_tk:.4f} ms")
    print(f"  加速比:   {fp16_time / total_int8_tk:.2f}x")
    
    # 完整验证
    print("\n" + "-" * 70)
    print("5. 完整流程验证 (包含所有操作)")
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
    
    full_pt_time = benchmark(full_int8_per_tensor)
    full_tk_time = benchmark(full_int8_per_token)
    
    print(f"FP16:              {fp16_time:.4f} ms (baseline)")
    print(f"INT8 Per-Tensor:   {full_pt_time:.4f} ms (加速 {fp16_time/full_pt_time:.2f}x)")
    print(f"INT8 Per-Token:    {full_tk_time:.4f} ms (加速 {fp16_time/full_tk_time:.2f}x)")
    
    # 不同的 M 值测试
    print("\n" + "-" * 70)
    print("6. 不同 batch*seq_len 的测试 (K=4096, N=4096)")
    print("-" * 70)
    
    print(f"\n{'M (B*S)':<15} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'Full INT8':<12} {'Speedup':<10}")
    print("-" * 65)
    
    test_Ms = [
        (1, "1"),
        (64, "64"),
        (256, "256"),
        (512, "512"),
        (1024, "1K"),
        (2048, "2K"),
        (4096, "4K"),
        (8192, "8K"),
        (16384, "16K"),
    ]
    
    for M_test, M_name in test_Ms:
        try:
            A_t = torch.randn(M_test, K, dtype=torch.float16, device=device)
            B_t = B_fp16  # 复用
            
            # 预量化
            A_s = A_t.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
            A_i = (A_t / A_s).round().clamp(-128, 127).to(torch.int8)
            
            # FP16
            t_fp16 = benchmark(lambda: torch.matmul(A_t, B_t), num_warmup=10, num_runs=50)
            
            # INT8 only
            t_int8 = benchmark(lambda: torch._int_mm(A_i, B_int8), num_warmup=10, num_runs=50)
            
            # Full INT8
            def full_int8():
                a_s = A_t.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
                a_i = (A_t / a_s).round().clamp(-128, 127).to(torch.int8)
                c = torch._int_mm(a_i, B_int8)
                return c.half() * a_s * B_scale
            
            t_full = benchmark(full_int8, num_warmup=10, num_runs=50)
            
            speedup = t_fp16 / t_int8
            
            print(f"{M_name:<15} {t_fp16:<12.4f} {t_int8:<12.4f} {t_full:<12.4f} {speedup:<10.2f}x")
            
            del A_t, A_i
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{M_name:<15} ERROR: {e}")
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"""
对于 {M} x {K} @ {K} x {N} 矩阵乘法:

- FP16 matmul:     {fp16_time:.4f} ms
- INT8 _int_mm:    {int8_time:.4f} ms  (纯计算加速 {fp16_time/int8_time:.2f}x)
- INT8 端到端:     {full_tk_time:.4f} ms (含量化/反量化, 加速 {fp16_time/full_tk_time:.2f}x)

结论:
- 纯 INT8 GEMM 加速: {fp16_time/int8_time:.2f}x
- 端到端 (含量化开销) 加速: {fp16_time/full_tk_time:.2f}x
""")


if __name__ == "__main__":
    main()
