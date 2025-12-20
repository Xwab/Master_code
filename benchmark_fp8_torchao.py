"""
FP8 vs FP16 矩阵乘法性能对比 - 使用 torchao

安装: pip install torchao
"""

import torch
import torch.nn as nn
import time
import sys

def check_environment():
    """检查环境."""
    print("=" * 70)
    print("环境检查")
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
    
    if cap < (8, 9):
        print(f"⚠️ FP8 需要 Ada Lovelace (sm_89+)")
    else:
        print("✓ GPU 支持 FP8")
    
    # 检查 torchao
    try:
        import torchao
        print(f"torchao version: {torchao.__version__}")
        print("✓ torchao 可用")
        return True
    except ImportError:
        print("❌ torchao 未安装")
        print("   安装: pip install torchao")
        return False


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


def run_torchao_fp8_test():
    """测试 torchao 的 FP8 功能."""
    print("\n" + "=" * 70)
    print("torchao FP8 功能测试")
    print("=" * 70)
    
    import torchao
    
    # 检查可用的量化方法
    print("\n可用的量化方法:")
    
    # 检查 float8 模块
    try:
        from torchao.float8 import Float8Linear
        print("  ✓ torchao.float8.Float8Linear")
        has_float8_linear = True
    except ImportError:
        print("  ✗ torchao.float8.Float8Linear 不可用")
        has_float8_linear = False
    
    # 检查量化 API
    try:
        from torchao.quantization import quantize_, float8_weight_only, float8_dynamic_activation_float8_weight
        print("  ✓ torchao.quantization.float8_weight_only")
        print("  ✓ torchao.quantization.float8_dynamic_activation_float8_weight")
        has_quant_api = True
    except ImportError:
        print("  ✗ torchao.quantization float8 API 不可用")
        has_quant_api = False
    
    # 检查 float8 实验性 API
    try:
        from torchao.float8.float8_linear import Float8Linear
        print("  ✓ torchao.float8.float8_linear.Float8Linear")
    except:
        pass
    
    return has_float8_linear or has_quant_api


def run_linear_benchmark():
    """使用 nn.Linear 进行 FP8 vs FP16 对比."""
    print("\n" + "=" * 70)
    print("nn.Linear FP8 vs FP16 对比")
    print("=" * 70)
    
    device = 'cuda'
    
    try:
        from torchao.quantization import quantize_, float8_weight_only, float8_dynamic_activation_float8_weight
        has_quant = True
    except ImportError:
        has_quant = False
        print("torchao quantization API 不可用，跳过此测试")
        return
    
    # 测试配置: (batch*seq, in_features, out_features, description)
    configs = [
        (4096, 4096, 4096, "4K x 4K -> 4K"),
        (16384, 4096, 4096, "16K x 4K -> 4K"),
        (65536, 4096, 4096, "64K x 4K -> 4K"),
        (262144, 4096, 4096, "262K x 4K -> 4K (B64xS4096)"),
        # Value reconstruction
        (4096, 256, 1024, "4K x 256 -> 1K"),
        (16384, 256, 1024, "16K x 256 -> 1K"),
        (65536, 256, 1024, "64K x 256 -> 1K"),
        (262144, 256, 1024, "262K x 256 -> 1K"),
        (262144, 256, 4096, "262K x 256 -> 4K"),
        (262144, 512, 4096, "262K x 512 -> 4K"),
    ]
    
    print(f"\n{'配置':<30} {'FP16 (ms)':<12} {'FP8 W-Only':<12} {'FP8 W8A8':<12} {'W-Only Spd':<10} {'W8A8 Spd':<10}")
    print("-" * 100)
    
    for M, K, N, desc in configs:
        try:
            # 创建输入
            x = torch.randn(M, K, dtype=torch.float16, device=device)
            
            # FP16 Linear
            linear_fp16 = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
            
            # FP8 Weight-Only Linear
            linear_fp8_wo = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
            linear_fp8_wo.weight.data = linear_fp16.weight.data.clone()
            try:
                quantize_(linear_fp8_wo, float8_weight_only())
                has_wo = True
            except Exception as e:
                has_wo = False
            
            # FP8 W8A8 Linear
            linear_fp8_w8a8 = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
            linear_fp8_w8a8.weight.data = linear_fp16.weight.data.clone()
            try:
                quantize_(linear_fp8_w8a8, float8_dynamic_activation_float8_weight())
                has_w8a8 = True
            except Exception as e:
                has_w8a8 = False
            
            # Benchmark FP16
            fp16_time = benchmark(lambda: linear_fp16(x), 10, 50)
            
            # Benchmark FP8 Weight-Only
            if has_wo:
                wo_time = benchmark(lambda: linear_fp8_wo(x), 10, 50)
                wo_spd = fp16_time / wo_time
            else:
                wo_time = float('nan')
                wo_spd = float('nan')
            
            # Benchmark FP8 W8A8
            if has_w8a8:
                w8a8_time = benchmark(lambda: linear_fp8_w8a8(x), 10, 50)
                w8a8_spd = fp16_time / w8a8_time
            else:
                w8a8_time = float('nan')
                w8a8_spd = float('nan')
            
            print(f"{desc:<30} {fp16_time:<12.4f} {wo_time:<12.4f} {w8a8_time:<12.4f} {wo_spd:<10.2f}x {w8a8_spd:<10.2f}x")
            
            del linear_fp16, linear_fp8_wo, linear_fp8_w8a8, x
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{desc:<30} OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{desc:<30} ERROR: {e}")


def run_float8_linear_benchmark():
    """使用 Float8Linear 进行测试."""
    print("\n" + "=" * 70)
    print("Float8Linear 直接测试")
    print("=" * 70)
    
    device = 'cuda'
    
    try:
        from torchao.float8 import Float8Linear
        from torchao.float8 import convert_to_float8_training
    except ImportError:
        try:
            from torchao.float8.float8_linear import Float8Linear
        except ImportError:
            print("Float8Linear 不可用，跳过此测试")
            return
    
    # 测试配置
    configs = [
        (4096, 4096, 4096, "4K x 4K -> 4K"),
        (16384, 4096, 4096, "16K x 4K -> 4K"),
        (65536, 4096, 4096, "64K x 4K -> 4K"),
        (262144, 4096, 4096, "262K x 4K -> 4K"),
    ]
    
    print(f"\n{'配置':<25} {'FP16 (ms)':<12} {'Float8 (ms)':<12} {'Speedup':<10}")
    print("-" * 65)
    
    for M, K, N, desc in configs:
        try:
            x = torch.randn(M, K, dtype=torch.float16, device=device)
            
            # FP16
            linear_fp16 = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
            
            # Float8Linear
            linear_fp8 = Float8Linear.from_float(
                nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
            )
            
            fp16_time = benchmark(lambda: linear_fp16(x), 10, 50)
            fp8_time = benchmark(lambda: linear_fp8(x), 10, 50)
            
            speedup = fp16_time / fp8_time
            print(f"{desc:<25} {fp16_time:<12.4f} {fp8_time:<12.4f} {speedup:<10.2f}x")
            
            del linear_fp16, linear_fp8, x
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{desc:<25} OOM")
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")


def run_manual_fp8_matmul():
    """手动 FP8 矩阵乘法测试."""
    print("\n" + "=" * 70)
    print("手动 FP8 矩阵乘法 (torchao.float8)")
    print("=" * 70)
    
    device = 'cuda'
    
    try:
        from torchao.float8.float8_tensor import Float8Tensor, ScaledMMConfig
        from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
        has_manual_fp8 = True
    except ImportError:
        try:
            # 尝试其他导入路径
            from torchao.float8 import Float8Tensor
            has_manual_fp8 = True
        except:
            has_manual_fp8 = False
    
    if not has_manual_fp8:
        print("手动 FP8 API 不可用，使用 torch 原生 FP8")
        run_torch_native_fp8()
        return
    
    print("使用 torchao Float8Tensor...")
    # 这里添加使用 Float8Tensor 的代码


def run_torch_native_fp8():
    """使用 torch 原生 FP8 (如果 _scaled_mm 可用)."""
    print("\n" + "=" * 70)
    print("torch 原生 FP8 测试")
    print("=" * 70)
    
    device = 'cuda'
    
    if not hasattr(torch, 'float8_e4m3fn'):
        print("torch.float8_e4m3fn 不可用")
        return
    
    # 测试 _scaled_mm
    print("\n测试 torch._scaled_mm...")
    
    M, K, N = 64, 128, 256
    
    try:
        # 创建 FP8 张量
        A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
        B_fp16 = torch.randn(N, K, dtype=torch.float16, device=device)  # 注意: (N, K)
        
        # 量化到 FP8
        A_fp8 = A_fp16.to(torch.float8_e4m3fn)
        B_fp8 = B_fp16.to(torch.float8_e4m3fn)
        
        # B 需要是 column-major，即 B.t() 是 row-major
        # 所以我们传入 B_fp8.t()，它的 shape 是 (K, N)
        
        scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
        scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # 尝试调用
        C = torch._scaled_mm(A_fp8, B_fp8.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
        print(f"✓ _scaled_mm 成功! 输出 shape: {C.shape}")
        
        # 运行基准测试
        run_scaled_mm_benchmark()
        
    except Exception as e:
        print(f"✗ _scaled_mm 失败: {e}")
        print("\n尝试其他方法...")


def run_scaled_mm_benchmark():
    """运行 _scaled_mm 基准测试."""
    print("\n" + "=" * 70)
    print("torch._scaled_mm 基准测试")
    print("=" * 70)
    
    device = 'cuda'
    
    configs = [
        (4096, 4096, 4096, "4K x 4K x 4K"),
        (16384, 4096, 4096, "16K x 4K x 4K"),
        (65536, 4096, 4096, "64K x 4K x 4K"),
        (262144, 4096, 4096, "262K x 4K x 4K"),
        (262144, 256, 1024, "262K x 256 -> 1K"),
        (262144, 256, 4096, "262K x 256 -> 4K"),
    ]
    
    print(f"\n{'配置':<25} {'FP16 (ms)':<12} {'FP8 (ms)':<12} {'Speedup':<10}")
    print("-" * 65)
    
    for M, K, N, desc in configs:
        try:
            # FP16
            A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
            B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
            
            fp16_time = benchmark(lambda: torch.matmul(A_fp16, B_fp16), 10, 50)
            
            # FP8
            A_fp8 = A_fp16.to(torch.float8_e4m3fn)
            # B 需要是 (N, K) 然后转置
            B_nk = torch.randn(N, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
            
            scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
            scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)
            
            def fp8_mm():
                return torch._scaled_mm(A_fp8, B_nk.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
            
            fp8_time = benchmark(fp8_mm, 10, 50)
            
            speedup = fp16_time / fp8_time
            print(f"{desc:<25} {fp16_time:<12.4f} {fp8_time:<12.4f} {speedup:<10.2f}x")
            
            del A_fp16, B_fp16, A_fp8, B_nk
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{desc:<25} OOM")
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")


def run_batch_seq_test():
    """不同 batch/seq 组合测试."""
    print("\n" + "=" * 70)
    print("不同 Batch/Seq 组合 (K=4096, N=4096)")
    print("=" * 70)
    
    device = 'cuda'
    K, N = 4096, 4096
    
    try:
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
        has_quant = True
    except:
        has_quant = False
        print("quantize_ 不可用")
        return
    
    # 预创建权重
    linear_fp16 = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
    
    linear_fp8 = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
    linear_fp8.weight.data = linear_fp16.weight.data.clone()
    
    try:
        quantize_(linear_fp8, float8_dynamic_activation_float8_weight())
    except Exception as e:
        print(f"量化失败: {e}")
        return
    
    print(f"\n{'Batch':<8} {'Seq':<8} {'M':<12} {'FP16 (ms)':<12} {'FP8 (ms)':<12} {'Speedup':<10}")
    print("-" * 70)
    
    configs = [
        (1, 4096),
        (4, 4096),
        (8, 4096),
        (16, 4096),
        (32, 4096),
        (64, 4096),
        (128, 4096),
        # 大 batch 短 seq
        (64, 128),
        (64, 256),
        (64, 512),
        (128, 128),
        (128, 256),
        (256, 128),
    ]
    
    for batch, seq in configs:
        try:
            M = batch * seq
            x = torch.randn(M, K, dtype=torch.float16, device=device)
            
            fp16_time = benchmark(lambda: linear_fp16(x), 10, 50)
            fp8_time = benchmark(lambda: linear_fp8(x), 10, 50)
            
            speedup = fp16_time / fp8_time
            print(f"{batch:<8} {seq:<8} {M:<12} {fp16_time:<12.4f} {fp8_time:<12.4f} {speedup:<10.2f}x")
            
            del x
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:<8} {seq:<8} OOM")
        except Exception as e:
            print(f"{batch:<8} {seq:<8} ERROR: {e}")
    
    del linear_fp16, linear_fp8
    torch.cuda.empty_cache()


def run_value_reconstruction_test():
    """Value 重建场景测试."""
    print("\n" + "=" * 70)
    print("Value 重建场景 (rank -> hidden)")
    print("=" * 70)
    
    device = 'cuda'
    
    try:
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
    except:
        print("quantize_ 不可用")
        return
    
    print(f"\n{'Batch':<8} {'Seq':<8} {'Rank':<8} {'Hidden':<8} {'FP16':<10} {'FP8':<10} {'Speedup':<10}")
    print("-" * 80)
    
    configs = [
        (64, 4096, 128, 1024),
        (64, 4096, 256, 1024),
        (64, 4096, 512, 1024),
        (64, 4096, 256, 4096),
        (128, 4096, 256, 1024),
        (128, 256, 256, 1024),
        (256, 128, 256, 1024),
        (256, 256, 256, 1024),
    ]
    
    for batch, seq, rank, hidden in configs:
        try:
            M = batch * seq
            K = rank
            N = hidden
            
            x = torch.randn(M, K, dtype=torch.float16, device=device)
            
            linear_fp16 = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
            
            linear_fp8 = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
            linear_fp8.weight.data = linear_fp16.weight.data.clone()
            quantize_(linear_fp8, float8_dynamic_activation_float8_weight())
            
            fp16_time = benchmark(lambda: linear_fp16(x), 10, 50)
            fp8_time = benchmark(lambda: linear_fp8(x), 10, 50)
            
            speedup = fp16_time / fp8_time
            print(f"{batch:<8} {seq:<8} {rank:<8} {hidden:<8} {fp16_time:<10.4f} {fp8_time:<10.4f} {speedup:<10.2f}x")
            
            del linear_fp16, linear_fp8, x
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:<8} {seq:<8} {rank:<8} {hidden:<8} OOM")
        except Exception as e:
            print(f"{batch:<8} {seq:<8} {rank:<8} {hidden:<8} ERROR: {e}")


def run_int8_comparison():
    """与 INT8 对比."""
    print("\n" + "=" * 70)
    print("FP8 vs INT8 vs FP16 对比")
    print("=" * 70)
    
    device = 'cuda'
    
    try:
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, int8_weight_only
        has_int8 = True
    except:
        has_int8 = False
    
    if not has_int8:
        print("INT8 量化不可用")
        return
    
    configs = [
        (65536, 4096, 4096, "64K x 4K -> 4K"),
        (262144, 4096, 4096, "262K x 4K -> 4K"),
        (262144, 256, 1024, "262K x 256 -> 1K"),
    ]
    
    print(f"\n{'配置':<25} {'FP16':<10} {'FP8':<10} {'INT8':<10} {'FP8 Spd':<10} {'INT8 Spd':<10}")
    print("-" * 80)
    
    for M, K, N, desc in configs:
        try:
            x = torch.randn(M, K, dtype=torch.float16, device=device)
            
            # FP16
            linear_fp16 = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
            
            # FP8
            linear_fp8 = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
            linear_fp8.weight.data = linear_fp16.weight.data.clone()
            quantize_(linear_fp8, float8_dynamic_activation_float8_weight())
            
            # INT8
            linear_int8 = nn.Linear(K, N, bias=False, dtype=torch.float16, device=device)
            linear_int8.weight.data = linear_fp16.weight.data.clone()
            quantize_(linear_int8, int8_weight_only())
            
            fp16_time = benchmark(lambda: linear_fp16(x), 10, 50)
            fp8_time = benchmark(lambda: linear_fp8(x), 10, 50)
            int8_time = benchmark(lambda: linear_int8(x), 10, 50)
            
            fp8_spd = fp16_time / fp8_time
            int8_spd = fp16_time / int8_time
            
            print(f"{desc:<25} {fp16_time:<10.4f} {fp8_time:<10.4f} {int8_time:<10.4f} {fp8_spd:<10.2f}x {int8_spd:<10.2f}x")
            
            del linear_fp16, linear_fp8, linear_int8, x
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"{desc:<25} OOM")
        except Exception as e:
            print(f"{desc:<25} ERROR: {e}")


def main():
    if not check_environment():
        print("\n请安装 torchao: pip install torchao")
        sys.exit(1)
    
    # 测试 torchao FP8 功能
    if not run_torchao_fp8_test():
        print("\ntorchao FP8 功能不完整")
    
    # 运行各种测试
    run_linear_benchmark()
    run_batch_seq_test()
    run_value_reconstruction_test()
    run_int8_comparison()
    
    # 尝试 torch 原生 FP8
    run_torch_native_fp8()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
torchao FP8 使用说明:

1. 安装:
   pip install torchao

2. Weight-Only FP8:
   from torchao.quantization import quantize_, float8_weight_only
   quantize_(model, float8_weight_only())

3. W8A8 FP8 (动态激活量化):
   from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
   quantize_(model, float8_dynamic_activation_float8_weight())

4. FP8 优势:
   - 输出是 FP16，不像 INT8 输出 INT32
   - 更好的精度 (浮点量化)
   - Ada Lovelace+ 原生硬件支持

5. 与 INT8 对比:
   - FP8 精度更高
   - FP8 不需要 INT32->FP16 转换
   - 大矩阵时 FP8 通常更快
    """)


if __name__ == "__main__":
    main()
