"""
使用官方 KIVI 模型测试 PPL 的脚本

用法:
    python test_kivi_ppl.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --k_bits 2 \
        --v_bits 2 \
        --dataset wikitext2 \
        --seqlen 2048 \
        --limit 100
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

# 添加 KIVI 路径
KIVI_PATH = "/root/KIVI"
if os.path.exists(KIVI_PATH) and KIVI_PATH not in sys.path:
    sys.path.insert(0, KIVI_PATH)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Model path or name")
    parser.add_argument("--k_bits", type=int, default=2,
                        help="Key quantization bits")
    parser.add_argument("--v_bits", type=int, default=2,
                        help="Value quantization bits")
    parser.add_argument("--group_size", type=int, default=128,
                        help="Quantization group size")
    parser.add_argument("--residual_length", type=int, default=128,
                        help="Number of recent tokens to keep in full precision")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4", "ptb"],
                        help="Dataset for PPL evaluation")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of samples")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--baseline", action="store_true",
                        help="Also test baseline (no quantization)")
    parser.add_argument("--use_cache", action="store_true",
                        help="Use KV cache (prefill + decode mode)")
    parser.add_argument("--prefill_length", type=int, default=256,
                        help="Prefill length when using cache")
    return parser.parse_args()


def load_dataset(dataset_name, tokenizer):
    """加载数据集"""
    from datasets import load_dataset as hf_load_dataset
    
    print(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "wikitext2":
        testdata = hf_load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    elif dataset_name == "c4":
        testdata = hf_load_dataset(
            'allenai/c4', 'allenai--c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation'
        )
        testenc = tokenizer(' '.join(testdata[:1100]['text']), return_tensors='pt')
    elif dataset_name == "ptb":
        testdata = hf_load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return testenc




@torch.no_grad()
def eval_ppl_with_cache(
    model, 
    tokenizer,
    model_name: str,
    datasets: str = "wikitext2",
    seqlen: int = 2048, 
    limit: int = None, 
    device: str = "cuda", 
    prefill_length: int = 256, 
    group_size: int = 128
):
    """
    评估 PPL (使用 KV cache，触发 KIVI 量化)
    
    支持多个数据集和缓存的 testloader
    
    流程:
    1. Prefill: 一次性处理前 prefill_length 个 token
    2. Decode: 逐 token 处理剩余的 token
    
    Args:
        model: 模型
        tokenizer: tokenizer
        model_name: 模型名称 (用于缓存文件名)
        datasets: 数据集名称，逗号分隔 (如 "wikitext2,c4,ptb")
        seqlen: 序列长度
        limit: 最大样本数
        device: 设备
        prefill_length: prefill 阶段处理的 token 数
        group_size: 量化分组大小
    
    Returns:
        dict: {dataset_name: ppl}
    """
    model = model.to(device)
    model.eval()
    
    # 确保 prefill_length 对齐到 group_size
    prefill_length = ((prefill_length + group_size - 1) // group_size) * group_size
    
    results = {}
    
    for dataset in datasets.split(","):
        dataset = dataset.strip()
        
        # 加载或使用缓存的 testloader
        cache_testloader = (
            f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
        )
        
        if os.path.exists(cache_testloader):
            print(f"Loading cached testloader from {cache_testloader}")
            testloader = torch.load(cache_testloader)
        else:
            print(f"Creating testloader for {dataset}...")
            testloader = load_dataset(dataset, tokenizer)
            torch.save(testloader, cache_testloader)
            print(f"Saved to {cache_testloader}")
        
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        
        if limit is not None:
            nsamples = min(nsamples, limit)
        
        print(f"\nEvaluating {dataset}: {nsamples} samples, seqlen={seqlen}, prefill_length={prefill_length}")
        
        total_loss = 0.0
        total_tokens = 0
        
        for i in tqdm(range(nsamples), desc=f"Eval {dataset}"):
            batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(device)
            seq_len = batch.size(1)
            
            if seq_len <= prefill_length:
                # 序列太短，直接 forward
                outputs = model(batch, use_cache=False)
                logits = outputs.logits
                shift_logits = logits[:, :-1, :]
                shift_labels = batch[:, 1:]
                loss = nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
                continue
            
            past_key_values = None
            
            # ============================================================
            # 阶段 1: Prefill
            # ============================================================
            prefill_tokens = batch[:, :prefill_length]
            outputs = model(
                prefill_tokens,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits
            
            # Prefill 部分的 loss
            shift_logits = logits[:, :-1, :]
            shift_labels = prefill_tokens[:, 1:]
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
            
            # ============================================================
            # 阶段 2: Decode (逐 token)
            # ============================================================
            for j in range(prefill_length, seq_len):
                curr_token = batch[:, j:j+1]
                
                outputs = model(
                    curr_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                
                # 计算 loss (预测下一个 token)
                if j < seq_len - 1:
                    logits = outputs.logits[:, -1, :]
                    target = batch[:, j + 1]
                    loss = nn.functional.cross_entropy(logits, target, reduction='sum')
                    total_loss += loss.item()
                    total_tokens += target.numel()
            
            # 调试信息 (第一个 sample)
            if i == 0:
                print(f"\n[DEBUG] Sample 0 cache info:")
                print(f"  Cache type: {type(past_key_values)}")
                if hasattr(past_key_values, 'get_seq_length'):
                    print(f"  Cache seq_length: {past_key_values.get_seq_length()}")
        
        ppl = torch.exp(torch.tensor(total_loss / total_tokens))
        results[dataset] = ppl.item()
        print(f"\n{dataset} PPL: {ppl.item():.4f}")
    
    return results


@torch.no_grad()
def eval_ppl_simple(
    model, 
    tokenizer,
    model_name: str,
    datasets: str = "wikitext2",
    seqlen: int = 2048, 
    limit: int = None, 
    device: str = "cuda",
):
    """
    简单的 PPL 评估 (不使用 cache，直接 forward)
    
    支持多个数据集和缓存的 testloader
    """
    model = model.to(device)
    model.eval()
    
    results = {}
    loss_fct = nn.CrossEntropyLoss()
    
    for dataset in datasets.split(","):
        dataset = dataset.strip()
        
        # 加载或使用缓存的 testloader
        cache_testloader = (
            f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
        )
        
        if os.path.exists(cache_testloader):
            print(f"Loading cached testloader from {cache_testloader}")
            testloader = torch.load(cache_testloader)
        else:
            print(f"Creating testloader for {dataset}...")
            testloader = load_dataset(dataset, tokenizer)
            torch.save(testloader, cache_testloader)
            print(f"Saved to {cache_testloader}")
        
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        
        if limit is not None:
            nsamples = min(nsamples, limit)
        
        print(f"\nEvaluating {dataset}: {nsamples} samples, seqlen={seqlen}")
        
        nlls = []
        
        for i in tqdm(range(nsamples), desc=f"Eval {dataset}"):
            batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(device)
            
            outputs = model(batch, use_cache=False)
            logits = outputs.logits
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            nlls.append(loss.float() * (seqlen - 1))
        
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
        results[dataset] = ppl.item()
        print(f"\n{dataset} PPL: {ppl.item():.4f}")
    
    return results


def main():
    args = get_args()
    
    # 设置 dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    print("=" * 60)
    print("KIVI PPL Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"k_bits: {args.k_bits}")
    print(f"v_bits: {args.v_bits}")
    print(f"group_size: {args.group_size}")
    print(f"residual_length: {args.residual_length}")
    print(f"dataset: {args.dataset}")
    print(f"seqlen: {args.seqlen}")
    print(f"dtype: {args.dtype}")
    print("=" * 60)
    
    # 加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    results = {}
    
    # ================================================================
    # 测试 Baseline (可选)
    # ================================================================
    if args.baseline:
        print("\n" + "=" * 60)
        print("Testing Baseline (no quantization)")
        print("=" * 60)
        
        from transformers import LlamaForCausalLM
        
        baseline_model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if args.use_cache:
            baseline_results = eval_ppl_with_cache(
                baseline_model, tokenizer, args.model_path,
                datasets=args.dataset, seqlen=args.seqlen, limit=args.limit, 
                device=args.device, prefill_length=args.prefill_length, 
                group_size=args.group_size
            )
        else:
            baseline_results = eval_ppl_simple(
                baseline_model, tokenizer, args.model_path,
                datasets=args.dataset, seqlen=args.seqlen, limit=args.limit,
                device=args.device
            )
        
        results["baseline"] = baseline_results
        print(f"\nBaseline Results: {baseline_results}")
        
        # 释放内存
        del baseline_model
        torch.cuda.empty_cache()
    
    # ================================================================
    # 测试 KIVI
    # ================================================================
    print("\n" + "=" * 60)
    print(f"Testing KIVI (k_bits={args.k_bits}, v_bits={args.v_bits})")
    print("=" * 60)
    
    try:
        # 尝试导入 KIVI
        from KIVI.models.llama_kivi import LlamaForCausalLM_KIVI
        print("Successfully imported LlamaForCausalLM_KIVI")
    except ImportError as e:
        print(f"Error importing KIVI: {e}")
        print(f"Make sure KIVI is installed at {KIVI_PATH}")
        print("Trying alternative import...")
        
        try:
            from models.llama_kivi import LlamaForCausalLM_KIVI
            print("Successfully imported from models.llama_kivi")
        except ImportError:
            print("Failed to import KIVI. Please check your installation.")
            sys.exit(1)
    
    # 加载 KIVI 模型
    print(f"\nLoading KIVI model...")
    kivi_model = LlamaForCausalLM_KIVI.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
        residual_length=args.residual_length,
    )
    
    print(f"Model config:")
    print(f"  k_bits: {getattr(kivi_model.config, 'k_bits', 'N/A')}")
    print(f"  v_bits: {getattr(kivi_model.config, 'v_bits', 'N/A')}")
    print(f"  group_size: {getattr(kivi_model.config, 'group_size', 'N/A')}")
    print(f"  residual_length: {getattr(kivi_model.config, 'residual_length', 'N/A')}")
    
    if args.use_cache:
        print(f"\nUsing KV cache mode (prefill_length={args.prefill_length})")
        kivi_results = eval_ppl_with_cache(
            kivi_model, tokenizer, args.model_path,
            datasets=args.dataset, seqlen=args.seqlen, limit=args.limit,
            device=args.device, prefill_length=args.prefill_length,
            group_size=args.group_size
        )
    else:
        print(f"\nUsing direct forward mode (no cache)")
        kivi_results = eval_ppl_simple(
            kivi_model, tokenizer, args.model_path,
            datasets=args.dataset, seqlen=args.seqlen, limit=args.limit,
            device=args.device
        )
    
    results["kivi"] = kivi_results
    print(f"\nKIVI Results: {kivi_results}")
    
    # ================================================================
    # 结果汇总
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # 获取所有数据集
    all_datasets = [d.strip() for d in args.dataset.split(",")]
    
    # 打印表头
    header = f"{'Dataset':<15}"
    if "baseline" in results:
        header += f"{'Baseline':<15}"
    if "kivi" in results:
        header += f"{'KIVI':<15}"
    if "baseline" in results and "kivi" in results:
        header += f"{'Δ PPL':<15}{'Ratio':<15}"
    print(header)
    print("-" * len(header))
    
    # 打印每个数据集的结果
    for dataset in all_datasets:
        row = f"{dataset:<15}"
        
        baseline_ppl = results.get("baseline", {}).get(dataset, None)
        kivi_ppl = results.get("kivi", {}).get(dataset, None)
        
        if baseline_ppl is not None:
            row += f"{baseline_ppl:<15.4f}"
        if kivi_ppl is not None:
            row += f"{kivi_ppl:<15.4f}"
        if baseline_ppl is not None and kivi_ppl is not None:
            delta = kivi_ppl - baseline_ppl
            ratio = kivi_ppl / baseline_ppl
            row += f"{delta:+<15.4f}{ratio:<15.4f}"
        
        print(row)
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
