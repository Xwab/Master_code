"""
通用模型延时测试脚本

根据输入的模型名字加载本地模型，测试不同序列长度下的 prefill 和 decode 延时

使用方法:
    python benchmark_latency.py --models /path/to/model1,/path/to/model2
    python benchmark_latency.py --models meta-llama/Llama-2-7b-hf
    python benchmark_latency.py --models /path/to/model --seq_lens 512,1024,2048
"""

import torch
import time
import argparse
import gc
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache


# ============================================================================
# 配置
# ============================================================================

DEFAULT_SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DECODE_NEW_TOKENS = 32  # decode 阶段生成的 token 数


# ============================================================================
# 工具函数
# ============================================================================

def get_gpu_memory_info():
    """获取 GPU 内存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'total': total,
            'free': total - reserved,
        }
    return None


def print_gpu_info():
    """打印 GPU 信息"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA not available")


def clear_gpu_memory():
    """清理 GPU 内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# 模型加载
# ============================================================================

def load_model(model_path: str, device: str = "cuda", dtype = torch.float16, 
               use_flash_attn: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载模型和 tokenizer"""
    
    print(f"\nLoading model from: {model_path}")
    
    # 加载配置
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 确定 attention 实现
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 打印模型信息
    print(f"  Model type: {config.model_type}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    if hasattr(config, 'num_key_value_heads'):
        print(f"  Num KV heads: {config.num_key_value_heads}")
    print(f"  Max position: {getattr(config, 'max_position_embeddings', 'N/A')}")
    print(f"  Attention: {attn_impl}")
    print(f"  Dtype: {dtype}")
    
    mem = get_gpu_memory_info()
    if mem:
        print(f"  GPU Memory: {mem['allocated']:.2f} GB / {mem['total']:.1f} GB")
    
    return model, tokenizer


# ============================================================================
# Benchmark 函数
# ============================================================================

def benchmark_prefill(model, input_ids: torch.Tensor, warmup: int = 3, num_runs: int = 10) -> Dict:
    """
    测试 prefill 阶段的延时
    
    Args:
        model: 模型
        input_ids: 输入 token ids, shape: (1, seq_len)
        warmup: 预热次数
        num_runs: 测试次数
    
    Returns:
        包含延时统计的字典
    """
    seq_len = input_ids.shape[1]
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, use_cache=True)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            outputs = model(input_ids, use_cache=True)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    # 统计
    times = sorted(times)
    avg_time = sum(times) / len(times)
    min_time = times[0]
    max_time = times[-1]
    median_time = times[len(times) // 2]
    
    # 吞吐量
    throughput = seq_len / (avg_time / 1000)  # tokens/s
    
    return {
        'avg_ms': avg_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'median_ms': median_time,
        'throughput': throughput,  # tokens/s
        'tokens_per_ms': seq_len / avg_time,
    }


def benchmark_decode(model, input_ids: torch.Tensor, num_new_tokens: int = 32,
                     warmup: int = 2, num_runs: int = 5) -> Dict:
    """
    测试 decode 阶段的延时
    
    Args:
        model: 模型
        input_ids: 输入 token ids, shape: (1, seq_len)
        num_new_tokens: 生成的新 token 数
        warmup: 预热次数
        num_runs: 测试次数
    
    Returns:
        包含延时统计的字典
    """
    device = input_ids.device
    vocab_size = model.config.vocab_size
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            past_key_values = DynamicCache()
            outputs = model(input_ids, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            
            next_token = torch.randint(0, vocab_size, (1, 1), device=device)
            for _ in range(num_new_tokens):
                outputs = model(next_token, use_cache=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                next_token = torch.randint(0, vocab_size, (1, 1), device=device)
    
    # Benchmark
    all_token_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            # 先做 prefill
            past_key_values = DynamicCache()
            outputs = model(input_ids, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            
            next_token = torch.randint(0, vocab_size, (1, 1), device=device)
            
            # 测量 decode
            token_times = []
            for _ in range(num_new_tokens):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                outputs = model(next_token, use_cache=True, past_key_values=past_key_values)
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                past_key_values = outputs.past_key_values
                next_token = torch.randint(0, vocab_size, (1, 1), device=device)
                token_times.append((end - start) * 1000)  # ms
            
            all_token_times.extend(token_times)
    
    # 统计
    times = sorted(all_token_times)
    avg_time = sum(times) / len(times)
    min_time = times[0]
    max_time = times[-1]
    median_time = times[len(times) // 2]
    
    # 吞吐量
    throughput = 1000 / avg_time  # tokens/s
    
    return {
        'avg_ms': avg_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'median_ms': median_time,
        'throughput': throughput,  # tokens/s
    }


def run_benchmark_for_model(model, tokenizer, seq_lens: List[int], 
                            decode_tokens: int = 32, device: str = "cuda") -> List[Dict]:
    """
    对一个模型运行所有序列长度的 benchmark
    
    Returns:
        结果列表
    """
    results = []
    vocab_size = model.config.vocab_size
    max_pos = getattr(model.config, 'max_position_embeddings', 32768)
    
    for seq_len in seq_lens:
        print(f"\n  Testing seq_len = {seq_len}...", end=" ", flush=True)
        
        # 检查是否超过模型最大长度
        if seq_len > max_pos:
            print(f"SKIP (exceeds max_position_embeddings={max_pos})")
            results.append({
                'seq_len': seq_len,
                'status': 'skipped',
                'reason': f'exceeds max_position_embeddings={max_pos}',
            })
            continue
        
        try:
            # 创建随机输入
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
            
            # 测试 prefill
            prefill_result = benchmark_prefill(model, input_ids)
            
            # 测试 decode
            decode_result = benchmark_decode(model, input_ids, num_new_tokens=decode_tokens)
            
            # 获取内存使用
            mem = get_gpu_memory_info()
            
            result = {
                'seq_len': seq_len,
                'status': 'success',
                'prefill': prefill_result,
                'decode': decode_result,
                'memory_gb': mem['max_allocated'] if mem else None,
            }
            results.append(result)
            
            print(f"Prefill: {prefill_result['avg_ms']:.2f}ms, "
                  f"Decode: {decode_result['avg_ms']:.2f}ms/token")
            
            # 清理
            del input_ids
            clear_gpu_memory()
            
        except torch.cuda.OutOfMemoryError:
            print("OOM!")
            results.append({
                'seq_len': seq_len,
                'status': 'oom',
            })
            clear_gpu_memory()
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'seq_len': seq_len,
                'status': 'error',
                'error': str(e),
            })
    
    return results


# ============================================================================
# 结果输出
# ============================================================================

def print_results_table(model_name: str, results: List[Dict]):
    """打印结果表格"""
    
    print(f"\n{'=' * 90}")
    print(f"Results for: {model_name}")
    print(f"{'=' * 90}")
    
    # Prefill 表格
    print(f"\n{'PREFILL':^90}")
    print(f"{'SeqLen':>10} {'Avg(ms)':>12} {'Min(ms)':>12} {'Max(ms)':>12} {'Throughput':>15} {'Memory(GB)':>12}")
    print("-" * 90)
    
    for r in results:
        if r['status'] == 'success':
            p = r['prefill']
            mem = r.get('memory_gb', 0) or 0
            print(f"{r['seq_len']:>10} {p['avg_ms']:>12.2f} {p['min_ms']:>12.2f} "
                  f"{p['max_ms']:>12.2f} {p['throughput']:>12.0f} tok/s {mem:>12.2f}")
        elif r['status'] == 'oom':
            print(f"{r['seq_len']:>10} {'OOM':>12} {'-':>12} {'-':>12} {'-':>15} {'-':>12}")
        elif r['status'] == 'skipped':
            print(f"{r['seq_len']:>10} {'SKIP':>12} {'-':>12} {'-':>12} {'-':>15} {'-':>12}")
        else:
            print(f"{r['seq_len']:>10} {'ERROR':>12} {'-':>12} {'-':>12} {'-':>15} {'-':>12}")
    
    # Decode 表格
    print(f"\n{'DECODE (per token)':^90}")
    print(f"{'SeqLen':>10} {'Avg(ms)':>12} {'Min(ms)':>12} {'Max(ms)':>12} {'Throughput':>15}")
    print("-" * 90)
    
    for r in results:
        if r['status'] == 'success':
            d = r['decode']
            print(f"{r['seq_len']:>10} {d['avg_ms']:>12.2f} {d['min_ms']:>12.2f} "
                  f"{d['max_ms']:>12.2f} {d['throughput']:>12.0f} tok/s")
        elif r['status'] == 'oom':
            print(f"{r['seq_len']:>10} {'OOM':>12} {'-':>12} {'-':>12} {'-':>15}")
        elif r['status'] == 'skipped':
            print(f"{r['seq_len']:>10} {'SKIP':>12} {'-':>12} {'-':>12} {'-':>15}")
        else:
            print(f"{r['seq_len']:>10} {'ERROR':>12} {'-':>12} {'-':>12} {'-':>15}")


def save_results(all_results: Dict, output_file: str):
    """保存结果到 JSON 文件"""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark model latency")
    parser.add_argument('--models', type=str, required=True,
                        help='Model paths, comma-separated (e.g., /path/model1,/path/model2)')
    parser.add_argument('--seq_lens', type=str, default=None,
                        help='Sequence lengths to test, comma-separated (default: 512,1024,2048,4096,8192,16384,32768)')
    parser.add_argument('--decode_tokens', type=int, default=32,
                        help='Number of tokens to generate in decode phase (default: 32)')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'],
                        help='Data type (default: float16)')
    parser.add_argument('--no_flash_attn', action='store_true',
                        help='Disable flash attention')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    args = parser.parse_args()
    
    # 解析模型列表
    model_paths = [p.strip() for p in args.models.split(',')]
    
    # 解析序列长度
    if args.seq_lens:
        seq_lens = [int(s.strip().replace('k', '000').replace('K', '000')) 
                    for s in args.seq_lens.split(',')]
    else:
        seq_lens = DEFAULT_SEQ_LENS
    
    # 数据类型
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # 打印配置
    print("=" * 90)
    print("LATENCY BENCHMARK")
    print("=" * 90)
    print(f"Models: {model_paths}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Decode tokens: {args.decode_tokens}")
    print(f"Dtype: {args.dtype}")
    print(f"Flash Attention: {not args.no_flash_attn}")
    print()
    print_gpu_info()
    print("=" * 90)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seq_lens': seq_lens,
            'decode_tokens': args.decode_tokens,
            'dtype': args.dtype,
            'flash_attn': not args.no_flash_attn,
        },
        'models': {},
    }
    
    # 依次测试每个模型
    for model_path in model_paths:
        print(f"\n{'#' * 90}")
        print(f"# Model: {model_path}")
        print(f"{'#' * 90}")
        
        try:
            # 加载模型
            clear_gpu_memory()
            model, tokenizer = load_model(
                model_path, 
                device=args.device, 
                dtype=dtype,
                use_flash_attn=not args.no_flash_attn
            )
            
            # 运行 benchmark
            results = run_benchmark_for_model(
                model, tokenizer, seq_lens, 
                decode_tokens=args.decode_tokens,
                device=args.device
            )
            
            # 打印结果
            print_results_table(model_path, results)
            
            # 保存结果
            all_results['models'][model_path] = results
            
            # 释放模型
            del model, tokenizer
            clear_gpu_memory()
            
        except Exception as e:
            print(f"Failed to load/benchmark model: {e}")
            import traceback
            traceback.print_exc()
            all_results['models'][model_path] = {'error': str(e)}
    
    # 保存结果到文件
    if args.output:
        save_results(all_results, args.output)
    else:
        # 默认保存到当前目录
        output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(all_results, output_file)
    
    # 如果有多个模型，打印对比
    if len(model_paths) > 1:
        print_comparison(all_results)


def print_comparison(all_results: Dict):
    """打印多模型对比"""
    
    print(f"\n{'=' * 90}")
    print("MODEL COMPARISON")
    print(f"{'=' * 90}")
    
    models = list(all_results['models'].keys())
    seq_lens = all_results['config']['seq_lens']
    
    # Prefill 对比
    print(f"\n{'PREFILL Avg Latency (ms)':^90}")
    header = f"{'SeqLen':>10}"
    for m in models:
        name = m.split('/')[-1][:20]
        header += f" {name:>15}"
    print(header)
    print("-" * (10 + 16 * len(models)))
    
    for seq_len in seq_lens:
        row = f"{seq_len:>10}"
        for m in models:
            results = all_results['models'].get(m, [])
            if isinstance(results, dict) and 'error' in results:
                row += f" {'ERROR':>15}"
            else:
                r = next((x for x in results if x.get('seq_len') == seq_len), None)
                if r and r.get('status') == 'success':
                    row += f" {r['prefill']['avg_ms']:>15.2f}"
                elif r and r.get('status') == 'oom':
                    row += f" {'OOM':>15}"
                else:
                    row += f" {'-':>15}"
        print(row)
    
    # Decode 对比
    print(f"\n{'DECODE Avg Latency (ms/token)':^90}")
    print(header)
    print("-" * (10 + 16 * len(models)))
    
    for seq_len in seq_lens:
        row = f"{seq_len:>10}"
        for m in models:
            results = all_results['models'].get(m, [])
            if isinstance(results, dict) and 'error' in results:
                row += f" {'ERROR':>15}"
            else:
                r = next((x for x in results if x.get('seq_len') == seq_len), None)
                if r and r.get('status') == 'success':
                    row += f" {r['decode']['avg_ms']:>15.2f}"
                elif r and r.get('status') == 'oom':
                    row += f" {'OOM':>15}"
                else:
                    row += f" {'-':>15}"
        print(row)


if __name__ == "__main__":
    main()
