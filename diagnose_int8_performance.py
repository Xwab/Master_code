"""
诊断 INT8 性能问题

分析为什么 torch._int_mm 没有比 FP16 快
"""

import torch
import torch.nn as nn
import time
import os

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


def check_environment():
    """检查环境配置."""
    print("=" * 70)
    print("1. 环境检查")
    print("=" * 70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        
        print(f"\nGPU: {gpu}")
        print(f"Compute Capability: {cap[0]}.{cap[1]}")
        print(f"SM Count: {props.multi_processor_count}")
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        
        # 架构信息
        arch_map = {
            (7, 0): ("Volta", "V100", "INT8 TC: 125 TOPS"),
            (7, 5): ("Turing", "RTX 20xx", "INT8 TC: ~130 TOPS"),
            (8, 0): ("Ampere", "A100", "INT8 TC: 624 TOPS"),
            (8, 6): ("Ampere", "RTX 30xx", "INT8 TC: ~300 TOPS"),
            (8, 9): ("Ada Lovelace", "RTX 40xx/L20", "INT8 TC: ~480 TOPS, FP8 原生"),
            (9, 0): ("Hopper", "H100", "INT8 TC: 1979 TOPS"),
        }
        
        if cap in arch_map:
            arch, example, perf = arch_map[cap]
            print(f"Architecture: {arch} ({example})")
            print(f"理论性能: {perf}")
        
        # 检查 TF32
        print(f"\nTF32 for matmul: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"TF32 for cuDNN: {torch.backends.cudnn.allow_tf32}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")


def check_int_mm_implementation():
    """检查 torch._int_mm 的实现."""
    print("\n" + "=" * 70)
    print("2. torch._int_mm 实现检查")
    print("=" * 70)
    
    if not hasattr(torch, '_int_mm'):
        print("❌ torch._int_mm 不可用!")
        return False
    
    print("✓ torch._int_mm 可用")
    
    # 测试基本功能
    A = torch.randint(-128, 127, (16, 16), dtype=torch.int8, device='cuda')
    B = torch.randint(-128, 127, (16, 16), dtype=torch.int8, device='cuda')
    
    try:
        C = torch._int_mm(A, B)
        print(f"✓ 基本测试通过")
        print(f"  输入: {A.dtype}, 输出: {C.dtype}")
        print(f"  注意: 输出是 INT32，需要额外转换!")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True


def analyze_memory_bandwidth():
    """分析内存带宽瓶颈."""
    print("\n" + "=" * 70)
    print("3. 内存带宽分析")
    print("=" * 70)
    
    device = 'cuda'
    
    # 测试不同大小的矩阵
    configs = [
        (1024, 1024, 1024, "小矩阵"),
        (4096, 4096, 4096, "中矩阵"),
        (16384, 4096, 4096, "大矩阵 (64x256 tokens)"),
        (262144, 4096, 4096, "超大矩阵 (64x4096 tokens)"),
    ]
    
    print(f"\n{'配置':<25} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'INT8/FP16':<12} {'理论加速':<10}")
    print("-" * 75)
    
    for M, K, N, desc in configs:
        try:
            # 计算理论值
            # FP16: 读 A(M*K*2) + B(K*N*2), 写 C(M*N*2)
            # INT8: 读 A(M*K*1) + B(K*N*1), 写 C(M*N*4) [INT32输出]
            
            fp16_bytes = M*K*2 + K*N*2 + M*N*2
            int8_bytes = M*K*1 + K*N*1 + M*N*4  # INT32 输出!
            
            # 计算 FLOPS
            flops = 2 * M * K * N
            
            # 算术强度 (FLOPS / Bytes)
            fp16_intensity = flops / fp16_bytes
            int8_intensity = flops / int8_bytes
            
            A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
            B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
            A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
            B_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
            
            fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 50)
            int8_time = benchmark(lambda: torch._int_mm(A_int8, B_int8), 10, 50)
            
            ratio = int8_time / fp16_time
            theoretical = 0.5  # INT8 理论上是 FP16 的 2x
            
            print(f"{desc:<25} {fp16_time:<12.4f} {int8_time:<12.4f} {ratio:<12.2f} {theoretical:<10}")
            
            del A_fp16, B_fp16, A_int8, B_int8
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{desc:<25} OOM")
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")
    
    print("\n分析:")
    print("  - INT8 输出是 INT32 (4 bytes)，而 FP16 输出是 2 bytes")
    print("  - 这意味着 INT8 写内存的带宽需求是 FP16 的 2x!")
    print("  - 对于内存带宽受限的场景，INT8 可能不会更快")


def test_output_dtype_impact():
    """测试输出数据类型的影响."""
    print("\n" + "=" * 70)
    print("4. INT32 输出的影响")
    print("=" * 70)
    
    device = 'cuda'
    M, K, N = 262144, 4096, 4096  # 64 * 4096
    
    A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    B_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
    
    # FP16 matmul
    fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 30)
    
    # INT8 matmul (输出 INT32)
    int8_time = benchmark(lambda: torch._int_mm(A_int8, B_int8), 10, 30)
    
    # INT8 matmul + 转换为 FP16
    def int8_with_convert():
        C_int32 = torch._int_mm(A_int8, B_int8)
        return C_int32.half()
    
    int8_convert_time = benchmark(int8_with_convert, 10, 30)
    
    # 预先计算结果，只测试转换时间
    C_int32 = torch._int_mm(A_int8, B_int8)
    convert_only_time = benchmark(lambda: C_int32.half(), 10, 30)
    
    print(f"\n输出大小: ({M}, {N})")
    print(f"FP16 输出: {M * N * 2 / 1e9:.2f} GB")
    print(f"INT32 输出: {M * N * 4 / 1e9:.2f} GB")
    
    print(f"\n{'操作':<30} {'时间 (ms)':<15}")
    print("-" * 50)
    print(f"{'FP16 matmul':<30} {fp16_time:<15.4f}")
    print(f"{'INT8 _int_mm':<30} {int8_time:<15.4f}")
    print(f"{'INT8 + 转FP16':<30} {int8_convert_time:<15.4f}")
    print(f"{'仅转换 INT32->FP16':<30} {convert_only_time:<15.4f}")
    
    print(f"\n结论:")
    print(f"  - INT8 计算本身: {int8_time:.4f} ms")
    print(f"  - 转换开销: {convert_only_time:.4f} ms")
    print(f"  - 总计: {int8_time + convert_only_time:.4f} ms vs FP16: {fp16_time:.4f} ms")
    
    del A_fp16, B_fp16, A_int8, B_int8, C_int32
    torch.cuda.empty_cache()


def test_cublas_config():
    """测试不同的 cuBLAS 配置."""
    print("\n" + "=" * 70)
    print("5. cuBLAS/cuBLASLt 配置测试")
    print("=" * 70)
    
    device = 'cuda'
    M, K, N = 262144, 4096, 4096
    
    A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    print("\n测试不同 TF32 设置对 FP16 的影响:")
    
    # 禁用 TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fp16_no_tf32 = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 30)
    
    # 启用 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    fp16_tf32 = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 30)
    
    print(f"  FP16 (TF32 禁用): {fp16_no_tf32:.4f} ms")
    print(f"  FP16 (TF32 启用): {fp16_tf32:.4f} ms")
    print(f"  TF32 加速: {fp16_no_tf32 / fp16_tf32:.2f}x")
    
    print("\n注意: TF32 使用 Tensor Core 加速 FP32/FP16 计算")
    print("这可能使 FP16 已经非常快，INT8 难以超越")
    
    del A_fp16, B_fp16
    torch.cuda.empty_cache()


def test_torch_compile():
    """测试 torch.compile 的影响."""
    print("\n" + "=" * 70)
    print("6. torch.compile 优化测试")
    print("=" * 70)
    
    device = 'cuda'
    M, K, N = 262144, 4096, 4096
    
    A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    B_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
    
    # 普通版本
    def fp16_matmul():
        return torch.matmul(A_fp16, B_fp16)
    
    def int8_matmul():
        return torch._int_mm(A_int8, B_int8)
    
    fp16_time = benchmark(fp16_matmul, 10, 30)
    int8_time = benchmark(int8_matmul, 10, 30)
    
    print(f"\n普通版本:")
    print(f"  FP16: {fp16_time:.4f} ms")
    print(f"  INT8: {int8_time:.4f} ms")
    
    # torch.compile 版本
    try:
        fp16_compiled = torch.compile(fp16_matmul, mode="max-autotune")
        int8_compiled = torch.compile(int8_matmul, mode="max-autotune")
        
        # Warmup compile
        for _ in range(5):
            fp16_compiled()
            int8_compiled()
        torch.cuda.synchronize()
        
        fp16_compiled_time = benchmark(fp16_compiled, 10, 30)
        int8_compiled_time = benchmark(int8_compiled, 10, 30)
        
        print(f"\ntorch.compile (max-autotune):")
        print(f"  FP16: {fp16_compiled_time:.4f} ms")
        print(f"  INT8: {int8_compiled_time:.4f} ms")
        
    except Exception as e:
        print(f"\ntorch.compile 失败: {e}")
    
    del A_fp16, B_fp16, A_int8, B_int8
    torch.cuda.empty_cache()


def test_fp8_if_available():
    """测试 FP8 (如果可用)."""
    print("\n" + "=" * 70)
    print("7. FP8 测试 (Ada Lovelace 原生支持)")
    print("=" * 70)
    
    cap = torch.cuda.get_device_capability(0)
    
    if cap < (8, 9):
        print("FP8 需要 Ada Lovelace (sm_89) 或更高")
        return
    
    if not hasattr(torch, 'float8_e4m3fn'):
        print("当前 PyTorch 版本不支持 FP8")
        print("需要 PyTorch >= 2.1")
        return
    
    device = 'cuda'
    M, K, N = 262144, 4096, 4096
    
    try:
        A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
        B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
        
        # FP8 量化
        A_scale = A_fp16.abs().amax() / 448.0 + 1e-6  # FP8 E4M3 max
        B_scale = B_fp16.abs().amax() / 448.0 + 1e-6
        
        A_fp8 = (A_fp16 / A_scale).to(torch.float8_e4m3fn)
        B_fp8 = (B_fp16 / B_scale).to(torch.float8_e4m3fn)
        
        # FP16 baseline
        fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 30)
        
        # FP8 (如果有 _scaled_mm)
        if hasattr(torch, '_scaled_mm'):
            A_scale_t = torch.tensor(A_scale.item(), dtype=torch.float32, device=device)
            B_scale_t = torch.tensor(B_scale.item(), dtype=torch.float32, device=device)
            
            def fp8_matmul():
                return torch._scaled_mm(A_fp8, B_fp8, scale_a=A_scale_t, scale_b=B_scale_t, out_dtype=torch.float16)
            
            fp8_time = benchmark(fp8_matmul, 10, 30)
            
            print(f"\nFP16: {fp16_time:.4f} ms")
            print(f"FP8:  {fp8_time:.4f} ms")
            print(f"FP8 加速: {fp16_time / fp8_time:.2f}x")
        else:
            print("torch._scaled_mm 不可用")
        
        del A_fp16, B_fp16, A_fp8, B_fp8
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"FP8 测试失败: {e}")


def main():
    print("=" * 70)
    print("INT8 性能诊断工具")
    print("=" * 70)
    
    check_environment()
    
    if not check_int_mm_implementation():
        return
    
    analyze_memory_bandwidth()
    test_output_dtype_impact()
    test_cublas_config()
    test_torch_compile()
    test_fp8_if_available()
    
    print("\n" + "=" * 70)
    print("总结与建议")
    print("=" * 70)
    print("""
为什么 INT8 没有比 FP16 快？

1. **INT32 输出问题**:
   - torch._int_mm 输出是 INT32 (4 bytes)
   - FP16 输出是 2 bytes
   - 写内存带宽需求 INT8 是 FP16 的 2x!
   
2. **TF32 加速 FP16**:
   - 现代 GPU 的 FP16 使用 TF32 Tensor Core
   - 已经非常快，INT8 难以超越
   
3. **内存带宽瓶颈**:
   - 对于大矩阵，内存带宽是瓶颈
   - INT8 节省的计算被额外的内存访问抵消

4. **cuBLASLt 实现**:
   - PyTorch 的 _int_mm 可能没有完全优化
   - 专业框架 (TensorRT, vLLM) 有更好的实现

建议:
1. 对于 Ada Lovelace (L20), 尝试 FP8
2. 使用 TensorRT-LLM 或 vLLM 获得真正的加速
3. 如果只是为了节省显存，使用 Weight-Only INT8
4. 对于推理，考虑使用专门优化的 CUDA kernel
    """)


if __name__ == "__main__":
    main()
