"""
PPL (Perplexity) 评估脚本

支持 wikitext2, c4, ptb 数据集
支持有 KV cache 管理的模型（逐 token 生成）
支持 batch size > 1

使用方法:
    python eval_ppl.py --model /path/to/model --datasets wikitext2,c4,ptb
    python eval_ppl.py --model /path/to/model --batch_size 8 --seq_len 2048
    python eval_ppl.py --model /path/to/model --mode sequential --prefill_len 128
"""

import torch
import torch.nn.functional as F
import argparse
import time
import math
import gc
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset


# ============================================================================
# 配置
# ============================================================================

DEFAULT_PREFILL_LEN = 128
DEFAULT_SEQ_LEN = 2048
DEFAULT_BATCH_SIZE = 8


# ============================================================================
# 工具函数
# ============================================================================

def get_gpu_memory():
    """获取 GPU 内存使用"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def clear_gpu_memory():
    """清理 GPU 内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# 数据集加载
# ============================================================================

class TokenizerWrapper:
    """Wrapper for tokenized input IDs"""
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_ppl_eval_loaders(name: str, tokenizer, seqlen: int = 2048):
    """
    加载评估数据集
    
    Args:
        name: 数据集名称 - "wikitext2", "c4", "ptb"
        tokenizer: tokenizer
        seqlen: 序列长度
    
    Returns:
        TokenizerWrapper 或包含 input_ids 的对象
    """
    if "wikitext2" in name:
        print("Loading wikitext2...")
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        print(f"  Total tokens: {testenc.input_ids.shape[1]}")
        return testenc
    
    elif "c4" in name:
        print("Loading C4...")
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        testenc = TokenizerWrapper(testenc)
        print(f"  Total tokens: {testenc.input_ids.shape[1]}")
        return testenc
    
    elif "ptb" in name:
        print("Loading PTB...")
        valdata = load_dataset(
            "ptb-text-only/ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        print(f"  Total tokens: {testenc.input_ids.shape[1]}")
        return testenc
    
    else:
        raise NotImplementedError(f"Unknown dataset: {name}")


def split_into_sequences(input_ids: torch.Tensor, seq_len: int) -> List[torch.Tensor]:
    """
    将 tokenized input_ids 分割成固定长度的序列
    
    Args:
        input_ids: shape (1, total_len) 或 (total_len,)
        seq_len: 目标序列长度
    
    Returns:
        序列列表，每个 shape (seq_len,)
    """
    # 确保是 1D
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    
    total_len = input_ids.shape[0]
    
    # 计算可以分出多少完整序列
    num_sequences = total_len // seq_len
    
    sequences = []
    for i in range(num_sequences):
        start = i * seq_len
        end = start + seq_len
        sequences.append(input_ids[start:end])
    
    return sequences


def create_batches(sequences: List[torch.Tensor], batch_size: int) -> List[torch.Tensor]:
    """
    将序列分成 batches，处理不能整除的情况
    
    Args:
        sequences: 序列列表
        batch_size: batch 大小
    
    Returns:
        batch 列表，每个 shape (batch_size, seq_len) 或 (remaining, seq_len)
    """
    batches = []
    num_sequences = len(sequences)
    
    for i in range(0, num_sequences, batch_size):
        batch_seqs = sequences[i:i + batch_size]
        
        # 如果是最后一个 batch 且不满，仍然处理
        if len(batch_seqs) > 0:
            batch = torch.stack(batch_seqs)
            batches.append(batch)
    
    return batches


def prepare_dataset(name: str, tokenizer, seq_len: int, batch_size: int) -> Tuple[List[torch.Tensor], int]:
    """
    加载数据集并准备 batches
    
    Returns:
        (batches, num_sequences)
    """
    # 加载数据
    testenc = get_ppl_eval_loaders(name, tokenizer, seq_len)
    input_ids = testenc.input_ids
    
    # 分割成序列
    sequences = split_into_sequences(input_ids, seq_len)
    print(f"  Number of sequences: {len(sequences)}")
    
    if len(sequences) == 0:
        return [], 0
    
    # 创建 batches
    batches = create_batches(sequences, batch_size)
    print(f"  Number of batches: {len(batches)}")
    
    # 打印最后一个 batch 的信息
    if len(batches) > 0:
        last_batch_size = batches[-1].shape[0]
        if last_batch_size < batch_size:
            print(f"  Last batch size: {last_batch_size} (partial)")
    
    return batches, len(sequences)


# ============================================================================
# PPL 计算 - 标准方式
# ============================================================================

@torch.no_grad()
def compute_ppl_standard(model, batches: List[torch.Tensor], device: str = "cuda") -> Dict:
    """
    标准 PPL 计算（一次性前向传播）
    
    适用于没有特殊 KV cache 管理的模型
    """
    total_loss = 0.0
    total_tokens = 0
    
    model.eval()
    
    for batch in tqdm(batches, desc="Computing PPL (standard)"):
        batch = batch.to(device)
        batch_size, seq_len = batch.shape
        
        # 前向传播
        outputs = model(batch, use_cache=False)
        logits = outputs.logits
        
        # 计算 loss: 预测下一个 token
        # logits: (batch, seq_len, vocab_size)
        # targets: (batch, seq_len)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        # 计算交叉熵
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum'
        )
        
        total_loss += loss.item()
        total_tokens += batch_size * (seq_len - 1)
        
        # 清理
        del batch, outputs, logits
        clear_gpu_memory()
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    
    return {
        'ppl': ppl,
        'avg_loss': avg_loss,
        'total_tokens': total_tokens,
    }


# ============================================================================
# PPL 计算 - 逐 token 方式（适用于 KV cache 模型）
# ============================================================================

@torch.no_grad()
def compute_ppl_sequential(model, batches: List[torch.Tensor], 
                           prefill_len: int = 128, device: str = "cuda") -> Dict:
    """
    逐 token PPL 计算（适用于有 KV cache 管理的模型）
    
    流程：
    1. 先用 prefill_len 个 token 做 prefill
    2. 然后逐个 token 计算 loss
    
    Args:
        model: 模型
        batches: batch 列表，每个 batch shape: (batch_size, seq_len)
        prefill_len: prefill 阶段的 token 数
        device: 设备
    """
    total_loss = 0.0
    total_tokens = 0
    
    model.eval()
    vocab_size = model.config.vocab_size
    
    for batch in tqdm(batches, desc="Computing PPL (sequential)"):
        batch = batch.to(device)
        batch_size, seq_len = batch.shape
        
        # 确保 prefill_len 不超过 seq_len
        actual_prefill_len = min(prefill_len, seq_len - 1)
        
        # ===== Prefill 阶段 =====
        prefill_ids = batch[:, :actual_prefill_len]
        
        # Prefill 时计算 loss
        outputs = model(prefill_ids, use_cache=True)
        logits = outputs.logits  # (batch, prefill_len, vocab_size)
        past_key_values = outputs.past_key_values
        
        # Prefill 阶段的 loss (预测第 1 到 prefill_len 个 token)
        prefill_shift_logits = logits[:, :-1, :]  # (batch, prefill_len-1, vocab)
        prefill_shift_labels = batch[:, 1:actual_prefill_len]  # (batch, prefill_len-1)
        
        prefill_loss = F.cross_entropy(
            prefill_shift_logits.reshape(-1, vocab_size),
            prefill_shift_labels.reshape(-1),
            reduction='sum'
        )
        total_loss += prefill_loss.item()
        total_tokens += batch_size * (actual_prefill_len - 1)
        
        # ===== Decode 阶段（逐 token）=====
        # 从 prefill_len 开始，逐个计算
        for pos in range(actual_prefill_len, seq_len):
            # 当前输入 token
            current_token = batch[:, pos - 1:pos]  # (batch, 1)
            target_token = batch[:, pos]  # (batch,)
            
            # 前向传播
            outputs = model(
                current_token, 
                use_cache=True, 
                past_key_values=past_key_values
            )
            logits = outputs.logits  # (batch, 1, vocab_size)
            past_key_values = outputs.past_key_values
            
            # 计算 loss
            token_loss = F.cross_entropy(
                logits.squeeze(1),  # (batch, vocab_size)
                target_token,  # (batch,)
                reduction='sum'
            )
            
            total_loss += token_loss.item()
            total_tokens += batch_size
        
        # 清理模型 cache
        if hasattr(model, 'clear_cache'):
            model.clear_cache()
        elif hasattr(model, 'reset_cache'):
            model.reset_cache()
        
        del batch, outputs, past_key_values
        clear_gpu_memory()
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    
    return {
        'ppl': ppl,
        'avg_loss': avg_loss,
        'total_tokens': total_tokens,
    }


# ============================================================================
# PPL 计算 - 滑动窗口方式
# ============================================================================

@torch.no_grad()
def compute_ppl_sliding_window(model, batches: List[torch.Tensor], 
                                window_size: int = 512, stride: int = 256,
                                device: str = "cuda") -> Dict:
    """
    滑动窗口 PPL 计算
    
    对于很长的序列，使用滑动窗口来避免 OOM
    """
    total_loss = 0.0
    total_tokens = 0
    
    model.eval()
    vocab_size = model.config.vocab_size
    
    for batch in tqdm(batches, desc="Computing PPL (sliding window)"):
        batch = batch.to(device)
        batch_size, seq_len = batch.shape
        
        prev_end_loc = 0
        
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + window_size, seq_len)
            
            # 目标 token 的范围
            target_begin = max(begin_loc, prev_end_loc)
            target_len = end_loc - target_begin
            
            if target_len <= 0:
                continue
            
            # 输入序列
            input_ids = batch[:, begin_loc:end_loc]
            
            # 前向传播
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits
            
            # 只计算新增 token 的 loss
            # 在窗口内的相对位置
            relative_target_begin = target_begin - begin_loc
            
            shift_logits = logits[:, relative_target_begin:-1, :]
            shift_labels = batch[:, target_begin + 1:end_loc]
            
            if shift_labels.numel() > 0:
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, vocab_size),
                    shift_labels.reshape(-1),
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
            
            prev_end_loc = end_loc
            
            if end_loc >= seq_len:
                break
        
        del batch, outputs, logits
        clear_gpu_memory()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    ppl = math.exp(avg_loss) if total_tokens > 0 else float('inf')
    
    return {
        'ppl': ppl,
        'avg_loss': avg_loss,
        'total_tokens': total_tokens,
    }


# ============================================================================
# 模型加载
# ============================================================================

def load_model(model_path: str, device: str = "cuda", dtype = torch.float16,
               use_flash_attn: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载模型和 tokenizer"""
    
    print(f"\nLoading model from: {model_path}")
    
    # 确定 attention 实现
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Failed with flash_attention_2, trying without: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
    
    model.eval()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 打印模型信息
    config = model.config
    print(f"  Model type: {getattr(config, 'model_type', 'unknown')}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Vocab size: {config.vocab_size}")
    
    mem = get_gpu_memory()
    print(f"  GPU Memory: {mem:.2f} GB")
    
    return model, tokenizer


# ============================================================================
# 主评估函数
# ============================================================================

def evaluate_model(model, tokenizer, datasets: List[str], 
                   seq_len: int = 2048, batch_size: int = 8,
                   prefill_len: int = 128, mode: str = "auto",
                   device: str = "cuda") -> Dict:
    """
    评估模型在多个数据集上的 PPL
    
    Args:
        model: 模型
        tokenizer: tokenizer
        datasets: 数据集列表 ['wikitext2', 'ptb', 'c4']
        seq_len: 序列长度
        batch_size: batch size
        prefill_len: prefill 长度（用于 sequential 模式）
        mode: 计算模式 - "auto", "standard", "sequential", "sliding"
        device: 设备
    """
    results = {}
    
    # 自动检测模式
    if mode == "auto":
        # 检查是否有自定义 cache 管理
        has_custom_cache = (
            hasattr(model, 'clear_cache') or 
            hasattr(model, 'reset_cache') or
            hasattr(model, 'kv_cache')
        )
        if has_custom_cache:
            mode = "sequential"
            print(f"  Auto-detected: using sequential mode (custom KV cache)")
        else:
            mode = "standard"
            print(f"  Auto-detected: using standard mode")
    
    print(f"\n  Evaluation mode: {mode}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    if mode == "sequential":
        print(f"  Prefill length: {prefill_len}")
    
    # 支持的数据集
    supported_datasets = ['wikitext2', 'ptb', 'c4']
    
    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on: {dataset_name}")
        print(f"{'=' * 60}")
        
        # 检查是否支持
        if not any(ds in dataset_name for ds in supported_datasets):
            print(f"  Unknown dataset: {dataset_name}, skipping...")
            print(f"  Supported: {supported_datasets}")
            continue
        
        try:
            # 加载数据并准备 batches
            batches, num_sequences = prepare_dataset(dataset_name, tokenizer, seq_len, batch_size)
            
            if len(batches) == 0:
                print(f"  No sequences loaded, skipping...")
                continue
            
            # 计算 PPL
            start_time = time.time()
            
            if mode == "standard":
                result = compute_ppl_standard(model, batches, device)
            elif mode == "sequential":
                result = compute_ppl_sequential(model, batches, prefill_len, device)
            elif mode == "sliding":
                result = compute_ppl_sliding_window(model, batches, device=device)
            else:
                result = compute_ppl_standard(model, batches, device)
            
            elapsed = time.time() - start_time
            result['time_seconds'] = elapsed
            result['mode'] = mode
            result['num_sequences'] = num_sequences
            
            results[dataset_name] = result
            
            print(f"\n  Results for {dataset_name}:")
            print(f"    PPL: {result['ppl']:.4f}")
            print(f"    Avg Loss: {result['avg_loss']:.4f}")
            print(f"    Total Tokens: {result['total_tokens']}")
            print(f"    Sequences: {num_sequences}")
            print(f"    Time: {elapsed:.2f}s")
            
            # 清理
            del batches
            clear_gpu_memory()
            
        except Exception as e:
            print(f"  Error loading/evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {'error': str(e)}
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate model PPL on wikitext2/ptb")
    parser.add_argument('--model', type=str, required=True,
                        help='Model path or name')
    parser.add_argument('--datasets', type=str, default='wikitext2,c4,ptb',
                        help='Datasets to evaluate, comma-separated (default: wikitext2,c4,ptb)')
    parser.add_argument('--seq_len', type=int, default=DEFAULT_SEQ_LEN,
                        help=f'Sequence length (default: {DEFAULT_SEQ_LEN})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--prefill_len', type=int, default=DEFAULT_PREFILL_LEN,
                        help=f'Prefill length for sequential mode (default: {DEFAULT_PREFILL_LEN})')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'standard', 'sequential', 'sliding'],
                        help='PPL computation mode: auto, standard, sequential (for KV cache models), sliding')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Data type (default: float16)')
    parser.add_argument('--no_flash_attn', action='store_true',
                        help='Disable flash attention')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()
    
    # 解析数据集
    datasets = [d.strip() for d in args.datasets.split(',')]
    
    # 数据类型
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # 打印配置
    print("=" * 60)
    print("PPL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Datasets: {datasets}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Prefill length: {args.prefill_len}")
    print(f"Mode: {args.mode}")
    print(f"Dtype: {args.dtype}")
    print(f"Flash Attention: {not args.no_flash_attn}")
    print("=" * 60)
    
    # 加载模型
    model, tokenizer = load_model(
        args.model,
        device=args.device,
        dtype=dtype,
        use_flash_attn=not args.no_flash_attn
    )
    
    # 评估
    results = evaluate_model(
        model, tokenizer, datasets,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        prefill_len=args.prefill_len,
        mode=args.mode,
        device=args.device
    )
    
    # 汇总
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Dataset':<15} {'PPL':>12} {'Avg Loss':>12} {'Tokens':>12} {'Seqs':>8} {'Time (s)':>10}")
    print("-" * 70)
    
    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"{dataset_name:<15} {'ERROR':>12} {'-':>12} {'-':>12} {'-':>8} {'-':>10}")
        else:
            num_seqs = result.get('num_sequences', '-')
            print(f"{dataset_name:<15} {result['ppl']:>12.4f} {result['avg_loss']:>12.4f} "
                  f"{result['total_tokens']:>12} {num_seqs:>8} {result['time_seconds']:>10.2f}")
    
    # 保存结果
    if args.output:
        import json
        output_data = {
            'model': args.model,
            'config': {
                'seq_len': args.seq_len,
                'batch_size': args.batch_size,
                'prefill_len': args.prefill_len,
                'mode': args.mode,
                'dtype': args.dtype,
            },
            'results': results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # 返回结果供其他脚本使用
    return results


if __name__ == "__main__":
    main()
