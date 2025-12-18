"""
完整模型 Throughput 测试

对比 FP16 和 INT8 版本的低秩压缩模型:
- Prefill throughput (tokens/s)
- Decode throughput (tokens/s)
- Latency (ms/token)
- Memory usage

支持从硬盘加载已有模型权重
"""

import torch
import torch.nn as nn
import time
import argparse
import gc
from tqdm import tqdm
from typing import Optional, Dict, List
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig

# 导入两个版本的模型加载函数
from modeling_alrd_llama_fp16 import load_model_fp16
from modeling_alrd_llama_int8 import load_model_int8, print_int8_support


@dataclass
class BenchmarkResult:
    """Benchmark 结果"""
    model_type: str
    batch_size: int
    prefill_len: int
    decode_steps: int
    prefill_time_ms: float
    prefill_throughput: float
    decode_time_ms: float
    decode_throughput: float
    avg_decode_latency_ms: float
    peak_memory_gb: float


def get_truncation_ranks(config, rank_ratio: float = 0.25) -> Dict[str, int]:
    """
    生成 truncation_ranks 字典
    """
    ranks = {}
    num_layers = config.num_hidden_layers
    kv_dim = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
    rank = int(kv_dim * rank_ratio)
    
    for i in range(num_layers):
        ranks[f"model.layers.{i}.self_attn.k_proj"] = rank
        ranks[f"model.layers.{i}.self_attn.v_proj"] = rank
    
    return ranks


def warmup_model(model, input_ids, n_warmup: int = 3):
    """预热模型"""
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids, use_cache=True)
    torch.cuda.synchronize()


def benchmark_prefill(model, input_ids: torch.Tensor, n_iter: int = 10) -> tuple:
    """测试 Prefill 性能"""
    batch_size, seq_len = input_ids.shape
    
    warmup_model(model, input_ids)
    
    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_ids, use_cache=True)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    throughput = (batch_size * seq_len) / (avg_time / 1000)
    
    return avg_time, throughput


def benchmark_decode(model, prefill_ids: torch.Tensor, decode_steps: int = 32, n_iter: int = 5) -> tuple:
    """测试 Decode 性能"""
    batch_size = prefill_ids.shape[0]
    device = prefill_ids.device
    
    with torch.no_grad():
        outputs = model(prefill_ids, use_cache=True)
        past_key_values = outputs.past_key_values
    
    times = []
    
    with torch.no_grad():
        for _ in range(n_iter):
            outputs = model(prefill_ids, use_cache=True)
            past_kv = outputs.past_key_values
            
            next_token = torch.randint(0, 32000, (batch_size, 1), device=device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            for step in range(decode_steps):
                outputs = model(
                    next_token,
                    past_key_values=past_kv,
                    use_cache=True
                )
                past_kv = outputs.past_key_values
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    avg_total_time = sum(times) / len(times)
    throughput = (batch_size * decode_steps) / (avg_total_time / 1000)
    avg_latency = avg_total_time / decode_steps
    
    return avg_total_time, throughput, avg_latency


def get_peak_memory_gb() -> float:
    """获取峰值显存使用量 (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def run_benchmark(args):
    """运行完整 benchmark"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    truncation_ranks = get_truncation_ranks(config, args.rank_ratio)
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    prefill_lens = [int(x) for x in args.prefill_lens.split(',')]
    
    print("=" * 100)
    print("Model Throughput Benchmark: FP16 vs INT8 Value Reconstruction")
    print("=" * 100)
    print(f"Model: {args.model}")
    print(f"Rank ratio: {args.rank_ratio}")
    print(f"Decode steps: {args.decode_steps}")
    print(f"Iterations: {args.n_iter}")
    print("=" * 100)
    
    # 打印 INT8 支持情况
    print_int8_support()
    
    results = []
    
    for model_type in ["FP16", "INT8"]:
        print(f"\n{'='*50}")
        print(f"Loading {model_type} model...")
        print(f"{'='*50}")
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 加载模型
        if model_type == "FP16":
            model = load_model_fp16(args.model, truncation_ranks, device)
        else:
            model = load_model_int8(args.model, truncation_ranks, device)
        
        print(f"Model loaded. Peak memory: {get_peak_memory_gb():.2f} GB")
        
        for batch_size in batch_sizes:
            for prefill_len in prefill_lens:
                print(f"\n  Batch: {batch_size}, Prefill: {prefill_len}")
                
                input_ids = torch.randint(
                    0, config.vocab_size, 
                    (batch_size, prefill_len), 
                    device=device
                )
                
                torch.cuda.reset_peak_memory_stats()
                
                try:
                    prefill_time, prefill_throughput = benchmark_prefill(
                        model, input_ids, args.n_iter
                    )
                    
                    decode_time, decode_throughput, decode_latency = benchmark_decode(
                        model, input_ids, args.decode_steps, args.n_iter
                    )
                    
                    peak_memory = get_peak_memory_gb()
                    
                    result = BenchmarkResult(
                        model_type=model_type,
                        batch_size=batch_size,
                        prefill_len=prefill_len,
                        decode_steps=args.decode_steps,
                        prefill_time_ms=prefill_time,
                        prefill_throughput=prefill_throughput,
                        decode_time_ms=decode_time,
                        decode_throughput=decode_throughput,
                        avg_decode_latency_ms=decode_latency,
                        peak_memory_gb=peak_memory
                    )
                    results.append(result)
                    
                    print(f"    Prefill: {prefill_time:.2f}ms, {prefill_throughput:.0f} tokens/s")
                    print(f"    Decode:  {decode_time:.2f}ms, {decode_throughput:.0f} tokens/s, {decode_latency:.2f}ms/token")
                    print(f"    Memory:  {peak_memory:.2f} GB")
                    
                except Exception as e:
                    print(f"    Error: {e}")
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def print_comparison_table(results: List[BenchmarkResult]):
    """打印对比表格"""
    print("\n" + "=" * 120)
    print("Comparison Summary")
    print("=" * 120)
    
    configs = set((r.batch_size, r.prefill_len) for r in results)
    
    print(f"\n{'Batch':<8}{'Prefill':<10}{'Model':<8}"
          f"{'Prefill(ms)':<14}{'Prefill(t/s)':<14}"
          f"{'Decode(ms)':<12}{'Decode(t/s)':<14}{'Latency(ms)':<14}{'Memory(GB)':<12}")
    print("-" * 120)
    
    for batch_size, prefill_len in sorted(configs):
        fp16_results = [r for r in results 
                        if r.batch_size == batch_size 
                        and r.prefill_len == prefill_len 
                        and r.model_type == "FP16"]
        int8_results = [r for r in results 
                        if r.batch_size == batch_size 
                        and r.prefill_len == prefill_len 
                        and r.model_type == "INT8"]
        
        if fp16_results and int8_results:
            fp16 = fp16_results[0]
            int8 = int8_results[0]
            
            print(f"{batch_size:<8}{prefill_len:<10}{'FP16':<8}"
                  f"{fp16.prefill_time_ms:<14.2f}{fp16.prefill_throughput:<14.0f}"
                  f"{fp16.decode_time_ms:<12.2f}{fp16.decode_throughput:<14.0f}"
                  f"{fp16.avg_decode_latency_ms:<14.2f}{fp16.peak_memory_gb:<12.2f}")
            
            prefill_speedup = fp16.prefill_time_ms / int8.prefill_time_ms if int8.prefill_time_ms > 0 else 0
            decode_speedup = fp16.decode_time_ms / int8.decode_time_ms if int8.decode_time_ms > 0 else 0
            
            print(f"{'':<8}{'':<10}{'INT8':<8}"
                  f"{int8.prefill_time_ms:<14.2f}{int8.prefill_throughput:<14.0f}"
                  f"{int8.decode_time_ms:<12.2f}{int8.decode_throughput:<14.0f}"
                  f"{int8.avg_decode_latency_ms:<14.2f}{int8.peak_memory_gb:<12.2f}")
            
            print(f"{'':<8}{'':<10}{'Speedup':<8}"
                  f"{prefill_speedup:<14.2f}x{'':<14}"
                  f"{decode_speedup:<12.2f}x")
            print("-" * 120)


def main():
    parser = argparse.ArgumentParser(description='Model Throughput Benchmark')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True,
                        help='Model path (local path or HuggingFace model name)')
    parser.add_argument('--rank_ratio', type=float, default=0.25,
                        help='Low-rank ratio (0.25 = 25%% of original dim)')
    
    # 测试参数
    parser.add_argument('--batch_sizes', type=str, default='1,2,4',
                        help='Batch sizes to test (comma-separated)')
    parser.add_argument('--prefill_lens', type=str, default='128,512,1024',
                        help='Prefill lengths to test (comma-separated)')
    parser.add_argument('--decode_steps', type=int, default=32,
                        help='Number of decode steps')
    parser.add_argument('--n_iter', type=int, default=5,
                        help='Number of iterations per test')
    
    # 输出
    parser.add_argument('--output', type=str, default='',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    results = run_benchmark(args)
    
    if results:
        print_comparison_table(results)
    
    # 保存结果
    if args.output and results:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'model_type', 'batch_size', 'prefill_len', 'decode_steps',
                'prefill_time_ms', 'prefill_throughput', 
                'decode_time_ms', 'decode_throughput', 'avg_decode_latency_ms',
                'peak_memory_gb'
            ])
            for r in results:
                writer.writerow([
                    r.model_type, r.batch_size, r.prefill_len, r.decode_steps,
                    r.prefill_time_ms, r.prefill_throughput,
                    r.decode_time_ms, r.decode_throughput, r.avg_decode_latency_ms,
                    r.peak_memory_gb
                ])
        print(f"\nResults saved to {args.output}")
    
    print("\n" + "=" * 100)
    print("Benchmark Complete!")
    print("=" * 100)


if __name__ == '__main__':
    main()
