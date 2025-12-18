"""
Benchmark: INT8 matmul vs FP16 matmul for low-rank value reconstruction

场景: 低秩压缩后的 Value 重建
- Value latent: (batch_size, seq_len, rank) - token-wise 量化
- A矩阵 (重建矩阵): (head_dim * num_kv_heads, rank) - channel-wise 量化
- 重建: value_states = latent @ A^T

对比:
1. FP16 乘法: 直接 FP16 @ FP16
2. INT8 模拟乘法: 量化后模拟 INT8 @ INT8
3. torch.compile 优化版本

测试: 不同 batch_size, seq_len 下的 throughput
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from functools import partial
import sys

# ============================================================================
# 量化工具 (基于你的 quant_utils.py)
# ============================================================================

@torch.no_grad()
def quantize_symmetric(x, n_bits, dim=-1):
    """
    对称量化
    Args:
        x: 输入 tensor
        n_bits: 量化位数
        dim: 量化维度
    Returns:
        x_quant: 量化后的 tensor (仍为 float，但值为整数)
        scale: 缩放因子
    """
    q_max = 2**(n_bits - 1) - 1
    q_min = -2**(n_bits - 1)
    
    amax = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    scale = amax / q_max
    
    x_quant = (x / scale).round().clamp(q_min, q_max)
    return x_quant, scale


@torch.no_grad()
def quantize_to_int8(x, dim=-1):
    """量化到 INT8 并返回 int8 类型"""
    x_quant, scale = quantize_symmetric(x, n_bits=8, dim=dim)
    return x_quant.to(torch.int8), scale


@torch.no_grad()
def fake_quantize(x, n_bits, dim=-1):
    """Fake quantization: 量化后立即反量化"""
    x_quant, scale = quantize_symmetric(x, n_bits, dim)
    return x_quant * scale


# ============================================================================
# 乘法实现
# ============================================================================

class FP16Matmul(nn.Module):
    """标准 FP16 矩阵乘法"""
    def __init__(self, weight):
        super().__init__()
        # weight: (out_features, in_features)
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
    
    def forward(self, x):
        # x: (batch, seq, in_features)
        return torch.matmul(x, self.weight.T)


class INT8SimulatedMatmul(nn.Module):
    """
    模拟 INT8 矩阵乘法
    - x (Value latent): per-token 量化 (沿 dim=-1)
    - weight (A): per-channel 量化 (沿 dim=-1, 即每行独立量化)
    """
    def __init__(self, weight):
        super().__init__()
        # 预量化权重 (per-channel, 即每个输出通道独立量化)
        w_float = weight.float()
        w_int8, w_scale = quantize_to_int8(w_float, dim=-1)  # (out_features, in_features)
        self.w_int8 = nn.Parameter(w_int8, requires_grad=False)
        self.w_scale = nn.Parameter(w_scale, requires_grad=False)  # (out_features, 1)
    
    def forward(self, x):
        # x: (batch, seq, in_features)
        # Step 1: 量化输入 (per-token)
        x_int8, x_scale = quantize_to_int8(x, dim=-1)  # x_scale: (batch, seq, 1)
        
        # Step 2: INT8 矩阵乘法 (模拟，先转 int32)
        # x_int8: (batch, seq, in_features) @ w_int8.T: (in_features, out_features)
        out_int32 = torch.matmul(x_int8.int(), self.w_int8.T.int())
        
        # Step 3: 反量化
        # out = out_int32 * x_scale * w_scale.T
        out = out_int32.float() * x_scale * self.w_scale.T
        
        return out.to(x.dtype)


class INT8TrueMatmul(nn.Module):
    """
    使用 torch 原生 INT8 支持 (如果可用)
    在新版 PyTorch 中有 torch._int_mm 或 torch.nn.functional.scaled_dot_product_attention
    """
    def __init__(self, weight):
        super().__init__()
        w_float = weight.float()
        w_int8, w_scale = quantize_to_int8(w_float, dim=-1)
        
        # 权重需要转置并连续
        self.w_int8_T = nn.Parameter(w_int8.T.contiguous(), requires_grad=False)
        self.w_scale = nn.Parameter(w_scale, requires_grad=False)
        
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        
        # 检测可用的 INT8 实现
        self.has_int_mm = hasattr(torch, '_int_mm')
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 量化输入
        x_int8, x_scale = quantize_to_int8(x, dim=-1)
        
        if self.has_int_mm and x.device.type == 'cuda':
            # 使用 torch._int_mm (需要 2D 输入)
            x_2d = x_int8.view(-1, self.in_features).contiguous()
            try:
                out_int32 = torch._int_mm(x_2d, self.w_int8_T)
                out = out_int32.view(batch_size, seq_len, -1).float()
                out = out * x_scale * self.w_scale.T
                return out.to(x.dtype)
            except Exception as e:
                pass  # Fallback
        
        # Fallback: 使用 int32 模拟
        out_int32 = torch.matmul(x_int8.int(), self.w_int8_T.int())
        out = out_int32.float() * x_scale * self.w_scale.T
        return out.to(x.dtype)


class FakeQuantMatmul(nn.Module):
    """
    Fake Quantization: 量化-反量化后做 FP16 乘法
    用于比较量化误差，而非速度提升
    """
    def __init__(self, weight, x_bits=8, w_bits=8):
        super().__init__()
        self.x_bits = x_bits
        self.w_bits = w_bits
        
        # 预量化权重
        w_quant = fake_quantize(weight.float(), w_bits, dim=-1)
        self.weight = nn.Parameter(w_quant.to(weight.dtype), requires_grad=False)
    
    def forward(self, x):
        # Fake quantize input
        x_quant = fake_quantize(x.float(), self.x_bits, dim=-1).to(x.dtype)
        return torch.matmul(x_quant, self.weight.T)


# ============================================================================
# 完整的低秩 Value 重建模块
# ============================================================================

class LowRankValueFP16(nn.Module):
    """FP16 版本的低秩 Value 投影"""
    def __init__(self, hidden_size, rank, out_features):
        super().__init__()
        self.BLinear = nn.Linear(hidden_size, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=False)
    
    def forward(self, x):
        latent = self.BLinear(x)
        return self.ALinear(latent)


class LowRankValueINT8(nn.Module):
    """INT8 版本的低秩 Value 投影 (只在重建阶段使用 INT8)"""
    def __init__(self, hidden_size, rank, out_features, quantize_latent=True):
        super().__init__()
        self.quantize_latent = quantize_latent
        self.BLinear = nn.Linear(hidden_size, rank, bias=False)
        
        # ALinear 权重预量化
        A_weight = torch.randn(out_features, rank)
        A_int8, A_scale = quantize_to_int8(A_weight, dim=-1)
        self.A_int8 = nn.Parameter(A_int8, requires_grad=False)
        self.A_scale = nn.Parameter(A_scale, requires_grad=False)
        self.rank = rank
    
    def forward(self, x):
        # Step 1: 低秩投影 (FP16)
        latent = self.BLinear(x)  # (batch, seq, rank)
        
        if self.quantize_latent:
            # Step 2: 量化 latent (per-token)
            latent_int8, latent_scale = quantize_to_int8(latent, dim=-1)
            
            # Step 3: INT8 matmul
            out_int32 = torch.matmul(latent_int8.int(), self.A_int8.T.int())
            
            # Step 4: 反量化
            out = out_int32.float() * latent_scale * self.A_scale.T
            return out.to(x.dtype)
        else:
            # 使用预量化权重但 FP16 latent
            A_dequant = self.A_int8.float() * self.A_scale
            return torch.matmul(latent, A_dequant.T)


class LowRankValueCached(nn.Module):
    """
    模拟带 KV Cache 的场景:
    - Prefill: 一次性计算所有 token
    - Decode: 每次只计算 1 个 token，但需要重建整个历史
    """
    def __init__(self, hidden_size, rank, out_features, use_int8=False):
        super().__init__()
        self.use_int8 = use_int8
        self.BLinear = nn.Linear(hidden_size, rank, bias=False)
        
        if use_int8:
            A_weight = torch.randn(out_features, rank)
            A_int8, A_scale = quantize_to_int8(A_weight, dim=-1)
            self.A_int8 = nn.Parameter(A_int8, requires_grad=False)
            self.A_scale = nn.Parameter(A_scale, requires_grad=False)
        else:
            self.ALinear = nn.Linear(rank, out_features, bias=False)
    
    def forward_prefill(self, x):
        """Prefill: x is (batch, seq, hidden)"""
        latent = self.BLinear(x)
        
        if self.use_int8:
            latent_int8, latent_scale = quantize_to_int8(latent, dim=-1)
            out_int32 = torch.matmul(latent_int8.int(), self.A_int8.T.int())
            out = out_int32.float() * latent_scale * self.A_scale.T
            return out.to(x.dtype), latent  # 返回 latent 用于缓存
        else:
            return self.ALinear(latent), latent
    
    def forward_decode(self, new_x, cached_latent):
        """
        Decode: 
        - new_x: (batch, 1, hidden) - 新 token
        - cached_latent: (batch, seq, rank) - 缓存的历史 latent
        返回重建的完整 value states
        """
        new_latent = self.BLinear(new_x)  # (batch, 1, rank)
        all_latent = torch.cat([cached_latent, new_latent], dim=1)  # (batch, seq+1, rank)
        
        if self.use_int8:
            latent_int8, latent_scale = quantize_to_int8(all_latent, dim=-1)
            out_int32 = torch.matmul(latent_int8.int(), self.A_int8.T.int())
            out = out_int32.float() * latent_scale * self.A_scale.T
            return out.to(new_x.dtype), all_latent
        else:
            return self.ALinear(all_latent), all_latent


# ============================================================================
# Benchmark 函数
# ============================================================================

def warmup(fn, n_warmup=10):
    """预热"""
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_fn(fn, n_iter=100, n_warmup=10):
    """测量函数执行时间"""
    warmup(fn, n_warmup=n_warmup)
    
    times = []
    for _ in range(n_iter):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    return avg, std


def compute_throughput(batch_size, seq_len, time_ms):
    """计算 throughput (tokens/s)"""
    tokens = batch_size * seq_len
    time_s = time_ms / 1000.0
    return tokens / time_s if time_s > 0 else 0


# ============================================================================
# 主测试函数
# ============================================================================

def run_matmul_benchmark(args):
    """测试纯矩阵乘法性能"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    print("=" * 80)
    print("Test 1: Pure Matmul Benchmark (latent @ A^T)")
    print("=" * 80)
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Rank: {args.rank}, Out features: {args.head_dim * args.num_kv_heads}")
    print("=" * 80)
    
    out_features = args.head_dim * args.num_kv_heads
    weight = torch.randn(out_features, args.rank, device=device, dtype=dtype)
    
    # 创建实现
    fp16_mm = FP16Matmul(weight).to(device)
    int8_sim_mm = INT8SimulatedMatmul(weight).to(device)
    int8_true_mm = INT8TrueMatmul(weight).to(device)
    fake_quant_mm = FakeQuantMatmul(weight, x_bits=8, w_bits=8).to(device)
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    seq_lens = [int(x) for x in args.seq_lens.split(',')]
    
    results = []
    
    print(f"\n{'Batch':<8}{'SeqLen':<10}{'FP16(ms)':<12}{'INT8-Sim(ms)':<14}{'INT8-True(ms)':<15}{'Speedup':<10}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            x = torch.randn(batch_size, seq_len, args.rank, device=device, dtype=dtype)
            
            fp16_time, fp16_std = benchmark_fn(lambda: fp16_mm(x), n_iter=args.n_iter)
            int8_sim_time, _ = benchmark_fn(lambda: int8_sim_mm(x), n_iter=args.n_iter)
            int8_true_time, _ = benchmark_fn(lambda: int8_true_mm(x), n_iter=args.n_iter)
            
            speedup_sim = fp16_time / int8_sim_time if int8_sim_time > 0 else 0
            speedup_true = fp16_time / int8_true_time if int8_true_time > 0 else 0
            
            print(f"{batch_size:<8}{seq_len:<10}{fp16_time:<12.4f}{int8_sim_time:<14.4f}{int8_true_time:<15.4f}{speedup_true:<10.2f}x")
            
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'fp16_ms': fp16_time,
                'int8_sim_ms': int8_sim_time,
                'int8_true_ms': int8_true_time,
                'speedup': speedup_true,
            })
    
    return results


def run_lowrank_benchmark(args):
    """测试完整低秩 Value 重建性能"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    print("\n" + "=" * 80)
    print("Test 2: Full Low-Rank Value Reconstruction (BLinear + ALinear)")
    print("=" * 80)
    print(f"Hidden size: {args.hidden_size}, Rank: {args.rank}")
    print("=" * 80)
    
    out_features = args.head_dim * args.num_kv_heads
    
    fp16_model = LowRankValueFP16(args.hidden_size, args.rank, out_features).to(device).to(dtype)
    int8_model = LowRankValueINT8(args.hidden_size, args.rank, out_features, quantize_latent=True).to(device).to(dtype)
    int8_no_quant_latent = LowRankValueINT8(args.hidden_size, args.rank, out_features, quantize_latent=False).to(device).to(dtype)
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    seq_lens = [int(x) for x in args.seq_lens.split(',')]
    
    print(f"\n{'Batch':<8}{'SeqLen':<10}{'FP16(ms)':<12}{'INT8(ms)':<12}{'INT8-NoQ(ms)':<14}{'Speedup':<10}")
    print("-" * 80)
    
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            x = torch.randn(batch_size, seq_len, args.hidden_size, device=device, dtype=dtype)
            
            fp16_time, _ = benchmark_fn(lambda: fp16_model(x), n_iter=args.n_iter)
            int8_time, _ = benchmark_fn(lambda: int8_model(x), n_iter=args.n_iter)
            int8_noq_time, _ = benchmark_fn(lambda: int8_no_quant_latent(x), n_iter=args.n_iter)
            
            speedup = fp16_time / int8_time if int8_time > 0 else 0
            
            print(f"{batch_size:<8}{seq_len:<10}{fp16_time:<12.4f}{int8_time:<12.4f}{int8_noq_time:<14.4f}{speedup:<10.2f}x")
            
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'fp16_ms': fp16_time,
                'int8_ms': int8_time,
                'int8_noq_ms': int8_noq_time,
                'speedup': speedup,
            })
    
    return results


def run_decode_benchmark(args):
    """测试 Decode 场景 (增量生成)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    print("\n" + "=" * 80)
    print("Test 3: Decode Scenario (KV Cache + Reconstruction)")
    print("=" * 80)
    print("每次添加 1 个新 token，重建完整 Value states")
    print("=" * 80)
    
    out_features = args.head_dim * args.num_kv_heads
    
    fp16_model = LowRankValueCached(args.hidden_size, args.rank, out_features, use_int8=False).to(device).to(dtype)
    int8_model = LowRankValueCached(args.hidden_size, args.rank, out_features, use_int8=True).to(device).to(dtype)
    
    batch_sizes = [1, 2, 4]  # Decode 通常 batch 较小
    cache_lens = [128, 512, 1024, 2048, 4096]  # 已缓存的序列长度
    
    print(f"\n{'Batch':<8}{'CacheLen':<12}{'FP16(ms)':<12}{'INT8(ms)':<12}{'Speedup':<10}{'Throughput':<15}")
    print("-" * 80)
    
    results = []
    
    for batch_size in batch_sizes:
        for cache_len in cache_lens:
            # 模拟已缓存的 latent
            cached_latent = torch.randn(batch_size, cache_len, args.rank, device=device, dtype=dtype)
            new_x = torch.randn(batch_size, 1, args.hidden_size, device=device, dtype=dtype)
            
            def fp16_decode():
                return fp16_model.forward_decode(new_x, cached_latent)
            
            def int8_decode():
                return int8_model.forward_decode(new_x, cached_latent)
            
            fp16_time, _ = benchmark_fn(fp16_decode, n_iter=args.n_iter)
            int8_time, _ = benchmark_fn(int8_decode, n_iter=args.n_iter)
            
            speedup = fp16_time / int8_time if int8_time > 0 else 0
            throughput = compute_throughput(batch_size, cache_len + 1, int8_time)
            
            print(f"{batch_size:<8}{cache_len:<12}{fp16_time:<12.4f}{int8_time:<12.4f}{speedup:<10.2f}x{throughput:<15.0f}")
            
            results.append({
                'batch_size': batch_size,
                'cache_len': cache_len,
                'fp16_ms': fp16_time,
                'int8_ms': int8_time,
                'speedup': speedup,
                'throughput': throughput,
            })
    
    return results


def run_accuracy_check(args):
    """检查 INT8 量化的精度损失"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    print("\n" + "=" * 80)
    print("Test 4: Accuracy Check (INT8 vs FP16)")
    print("=" * 80)
    
    out_features = args.head_dim * args.num_kv_heads
    weight = torch.randn(out_features, args.rank, device=device, dtype=dtype)
    
    fp16_mm = FP16Matmul(weight).to(device)
    int8_sim_mm = INT8SimulatedMatmul(weight).to(device)
    fake_quant_mm = FakeQuantMatmul(weight, x_bits=8, w_bits=8).to(device)
    
    # 测试不同输入
    for seq_len in [128, 1024]:
        x = torch.randn(1, seq_len, args.rank, device=device, dtype=dtype)
        
        with torch.no_grad():
            out_fp16 = fp16_mm(x)
            out_int8 = int8_sim_mm(x)
            out_fake = fake_quant_mm(x)
        
        # 计算误差
        mse_int8 = ((out_fp16 - out_int8) ** 2).mean().item()
        mse_fake = ((out_fp16 - out_fake) ** 2).mean().item()
        cos_sim_int8 = F.cosine_similarity(out_fp16.flatten(), out_int8.flatten(), dim=0).item()
        cos_sim_fake = F.cosine_similarity(out_fp16.flatten(), out_fake.flatten(), dim=0).item()
        
        print(f"\nSeq len: {seq_len}")
        print(f"  INT8 Simulated - MSE: {mse_int8:.6f}, Cosine Sim: {cos_sim_int8:.6f}")
        print(f"  Fake Quantized - MSE: {mse_fake:.6f}, Cosine Sim: {cos_sim_fake:.6f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark INT8 vs FP16 matmul for low-rank reconstruction')
    
    # 模型参数
    parser.add_argument('--rank', type=int, default=256, help='Low-rank dimension')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--num_kv_heads', type=int, default=8, help='Number of KV heads')
    parser.add_argument('--hidden_size', type=int, default=4096, help='Hidden size')
    
    # 测试参数
    parser.add_argument('--batch_sizes', type=str, default='1,2,4,8', 
                        help='Batch sizes to test (comma-separated)')
    parser.add_argument('--seq_lens', type=str, default='128,512,1024,2048,4096', 
                        help='Sequence lengths to test (comma-separated)')
    parser.add_argument('--n_iter', type=int, default=100, help='Number of iterations per test')
    
    # 测试选择
    parser.add_argument('--test_matmul', action='store_true', help='Run pure matmul benchmark')
    parser.add_argument('--test_lowrank', action='store_true', help='Run low-rank reconstruction benchmark')
    parser.add_argument('--test_decode', action='store_true', help='Run decode scenario benchmark')
    parser.add_argument('--test_accuracy', action='store_true', help='Run accuracy check')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    # 输出
    parser.add_argument('--output', type=str, default='', help='Output CSV file')
    
    args = parser.parse_args()
    
    # 如果没有指定测试，默认运行所有
    if not any([args.test_matmul, args.test_lowrank, args.test_decode, args.test_accuracy, args.all]):
        args.all = True
    
    all_results = {}
    
    if args.test_matmul or args.all:
        all_results['matmul'] = run_matmul_benchmark(args)
    
    if args.test_lowrank or args.all:
        all_results['lowrank'] = run_lowrank_benchmark(args)
    
    if args.test_decode or args.all:
        all_results['decode'] = run_decode_benchmark(args)
    
    if args.test_accuracy or args.all:
        run_accuracy_check(args)
    
    # 保存结果
    if args.output:
        import pandas as pd
        for name, results in all_results.items():
            if results:
                df = pd.DataFrame(results)
                output_file = args.output.replace('.csv', f'_{name}.csv')
                df.to_csv(output_file, index=False)
                print(f"\nResults saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
