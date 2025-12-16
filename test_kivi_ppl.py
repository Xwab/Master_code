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
def eval_ppl(model, testenc, seqlen, limit, device, use_kivi_cache=False):
    """评估 PPL (不使用 cache，直接 forward)"""
    model.eval()
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen
    
    if limit is not None:
        nsamples = min(nsamples, limit)
    
    print(f"Evaluating {nsamples} samples, seqlen={seqlen}")
    
    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    
    for i in tqdm(range(nsamples), desc="Evaluating"):
        batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(device)
        
        # 不使用 cache，直接一次 forward 整个序列
        # KIVI 在 attention 内部会对 K/V 进行量化
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
    return ppl.item()


@torch.no_grad()
def eval_ppl_with_cache(model, testenc, seqlen, limit, device, 
                         prefill_length=256, group_size=128):
    """
    评估 PPL (使用 KV cache，触发 KIVI 量化)
    
    流程:
    1. Prefill: 一次性处理前 prefill_length 个 token
    2. Decode: 逐 token 处理剩余的 token
    """
    model.eval()
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen
    
    if limit is not None:
        nsamples = min(nsamples, limit)
    
    # 确保 prefill_length 对齐到 group_size
    prefill_length = ((prefill_length + group_size - 1) // group_size) * group_size
    
    print(f"Evaluating {nsamples} samples, seqlen={seqlen}, prefill_length={prefill_length}")
    
    total_loss = 0.0
    total_tokens = 0
    
    for i in tqdm(range(nsamples), desc="Evaluating"):
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
            target_token = batch[:, j:j+1] if j < seq_len - 1 else None
            
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
    return ppl.item()


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
    
    # 加载数据集
    testenc = load_dataset(args.dataset, tokenizer)
    
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
            ppl_baseline = eval_ppl_with_cache(
                baseline_model, testenc, args.seqlen, args.limit, args.device,
                prefill_length=args.prefill_length, group_size=args.group_size
            )
        else:
            ppl_baseline = eval_ppl(baseline_model, testenc, args.seqlen, args.limit, args.device)
        
        results["baseline"] = ppl_baseline
        print(f"\nBaseline PPL: {ppl_baseline:.4f}")
        
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
        ppl_kivi = eval_ppl_with_cache(
            kivi_model, testenc, args.seqlen, args.limit, args.device,
            prefill_length=args.prefill_length, group_size=args.group_size
        )
    else:
        print(f"\nUsing direct forward mode (no cache)")
        ppl_kivi = eval_ppl(kivi_model, testenc, args.seqlen, args.limit, args.device)
    
    results["kivi"] = ppl_kivi
    print(f"\nKIVI PPL: {ppl_kivi:.4f}")
    
    # ================================================================
    # 结果汇总
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for name, ppl in results.items():
        print(f"{name}: {ppl:.4f}")
    
    if "baseline" in results and "kivi" in results:
        delta = results["kivi"] - results["baseline"]
        ratio = results["kivi"] / results["baseline"]
        print(f"\nΔ PPL: {delta:+.4f}")
        print(f"Ratio: {ratio:.4f}x")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
