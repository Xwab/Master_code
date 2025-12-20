"""
KIVI 量化 KV Cache 的 PPL 测试函数

关键：逐 token/分块 forward，让 KV cache 累积并触发 KIVI 量化
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Dict, Any


def get_ppl_eval_loaders(dataset_name, tokenizer, seqlen=2048):
    """加载 PPL 评估数据集"""
    from datasets import load_dataset
    
    if dataset_name == "wikitext2":
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    elif dataset_name == "c4":
        testdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        testenc = tokenizer(' '.join(testdata[:1100]['text']), return_tensors='pt')
    elif dataset_name == "ptb":
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return testenc


@torch.no_grad()
def eval_ppl_kivi(
    model, 
    tokenizer, 
    model_name: str, 
    datasets: str = "wikitext2",
    seqlen: int = 2048, 
    device: str = "cuda",
    limit: int = 512,
    chunk_size: int = 256,
    verbose: bool = True,
):
    """
    KIVI 量化 KV Cache 的 PPL 测试函数
    
    Args:
        model: KIVI 量化的模型
        tokenizer: tokenizer
        model_name: 模型名称（用于缓存）
        datasets: 数据集名称，逗号分隔
        seqlen: 每个 sample 的序列长度
        device: 设备
        limit: 最大测试样本数
        chunk_size: 每次 forward 的 token 数
                   必须 > residual_length 才能触发量化！
        verbose: 是否打印调试信息
    
    Returns:
        包含各数据集 PPL 的字典
    """
    model = model.to(device)
    if isinstance(device, str):
        device = torch.device(device)

    # 获取 KIVI 参数
    k_bits = getattr(model.config, 'k_bits', 16)
    v_bits = getattr(model.config, 'v_bits', 16)
    group_size = getattr(model.config, 'group_size', 128)
    residual_length = getattr(model.config, 'residual_length', 128)
    
    # chunk_size 必须:
    # 1. > residual_length 才能触发量化
    # 2. 是 group_size 的整数倍（KIVI triton kernel 要求）
    if chunk_size <= residual_length:
        chunk_size = residual_length + group_size
        if verbose:
            print(f"[INFO] Adjusted chunk_size to {chunk_size} (> residual_length)")
    
    if chunk_size % group_size != 0:
        chunk_size = ((chunk_size // group_size) + 1) * group_size
        if verbose:
            print(f"[INFO] Aligned chunk_size to group_size: {chunk_size}")
    
    if verbose:
        print("=" * 60)
        print("KIVI PPL Evaluation")
        print("=" * 60)
        print(f"k_bits: {k_bits}")
        print(f"v_bits: {v_bits}")
        print(f"group_size: {group_size}")
        print(f"residual_length: {residual_length}")
        print(f"chunk_size: {chunk_size}")
        print(f"seqlen: {seqlen}")
        print("=" * 60)

    results = {}

    for dataset in datasets.split(","):
        dataset = dataset.strip()
        cache_testloader = (
            f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
        )
        
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
            if verbose:
                print(f"Loaded cached testloader from {cache_testloader}")
        else:
            testloader = get_ppl_eval_loaders(dataset, tokenizer, seqlen)
            torch.save(testloader, cache_testloader)
            if verbose:
                print(f"Saved testloader to {cache_testloader}")
        
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        
        use_cache_orig = model.config.use_cache
        model.config.use_cache = True
        model.eval()

        nlls = []
        total_tokens = 0
        loss_fct = nn.CrossEntropyLoss(reduction='sum')

        if verbose:
            print(f"\nEvaluating {dataset}: {min(nsamples, limit)} samples, seqlen={seqlen}")

        for i in tqdm(range(min(nsamples, limit)), desc=f"Eval {dataset}"):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
            seq_len = batch.size(1)
            
            # 重置 KV cache
            past_key_values = None
            sample_loss = 0.0
            sample_tokens = 0
            
            # ============================================================
            # 分块 forward - 触发 KIVI 量化
            # 注意：每个 chunk 必须是 group_size 的整数倍
            # ============================================================
            
            # 计算可以完整分块的长度
            n_full_chunks = seq_len // chunk_size
            aligned_len = n_full_chunks * chunk_size
            remainder = seq_len - aligned_len
            
            # 处理完整的 chunks
            for start in range(0, aligned_len, chunk_size):
                end = start + chunk_size
                chunk = batch[:, start:end]
                chunk_len = chunk_size
                
                # Forward with KV cache
                outputs = model(
                    chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits  # [batch, chunk_len, vocab_size]
                
                # 计算 loss: 预测 chunk 内的下一个 token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = chunk[:, 1:].contiguous()
                
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                sample_loss += loss.item()
                sample_tokens += shift_labels.numel()
                
                # chunk 最后一个 token 预测下一个 token
                if end < seq_len:
                    next_token_logits = logits[:, -1:, :]
                    next_token_label = batch[:, end:end+1]
                    
                    loss = loss_fct(
                        next_token_logits.view(-1, next_token_logits.size(-1)),
                        next_token_label.view(-1)
                    )
                    sample_loss += loss.item()
                    sample_tokens += next_token_label.numel()
            
            # 处理剩余的 tokens（逐 token 处理，避免不对齐问题）
            if remainder > 0:
                for j in range(aligned_len, seq_len - 1):
                    curr_token = batch[:, j:j+1]
                    
                    outputs = model(
                        curr_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    
                    logits = outputs.logits[:, -1, :]
                    target = batch[:, j + 1]
                    
                    loss = F.cross_entropy(logits, target, reduction='sum')
                    sample_loss += loss.item()
                    sample_tokens += target.numel()
            
            nlls.append(sample_loss)
            total_tokens += sample_tokens
            
            # ============================================================
            # 调试信息：检查 KIVI 是否生效
            # ============================================================
            if i == 0 and verbose:
                print(f"\n[DEBUG] Sample 0 - Cache Analysis:")
                print(f"  Cache type: {type(past_key_values)}")
                
                # 检查是否是 KIVI cache
                if hasattr(past_key_values, 'get_seq_length'):
                    print(f"  get_seq_length(): {past_key_values.get_seq_length()}")
                
                if hasattr(past_key_values, '_cache'):
                    # KIVILatentCache 结构
                    if 0 in past_key_values._cache:
                        k_quant, k_res, v_quant, v_res = past_key_values._cache[0]
                        k_quant_len = k_quant.shape[1] if k_quant is not None else 0
                        k_res_len = k_res.shape[1] if k_res is not None else 0
                        v_quant_len = v_quant.shape[1] if v_quant is not None else 0
                        v_res_len = v_res.shape[1] if v_res is not None else 0
                        print(f"  Layer 0 Key:   quantized={k_quant_len}, residual={k_res_len}")
                        print(f"  Layer 0 Value: quantized={v_quant_len}, residual={v_res_len}")
                        
                        if k_quant_len > 0:
                            print(f"  ✓ KIVI quantization IS active!")
                        else:
                            print(f"  ✗ KIVI quantization NOT active (all in residual)")
                
                elif isinstance(past_key_values, tuple):
                    # 标准 cache (tuple of tuples)
                    print(f"  Standard tuple cache (NOT KIVI)")
                    print(f"  Layer 0 key shape: {past_key_values[0][0].shape}")
                    print(f"  ✗ KIVI quantization NOT active!")
                
                print()

        # 计算 PPL
        total_nll = sum(nlls)
        ppl = torch.exp(torch.tensor(total_nll / total_tokens))
        
        model.config.use_cache = use_cache_orig
        results[dataset] = ppl.item()
        
        if verbose:
            print(f"\n{dataset} PPL: {ppl.item():.4f}")
            print(f"Total tokens: {total_tokens}")

    return results


@torch.no_grad()
def eval_ppl_kivi_token_by_token(
    model, 
    tokenizer, 
    model_name: str, 
    datasets: str = "wikitext2",
    seqlen: int = 2048, 
    device: str = "cuda",
    limit: int = 50,
    verbose: bool = True,
    prefill_length: int = None,  # 首次 prefill 的长度，默认使用 residual_length
):
    """
    逐 token 的 KIVI PPL 测试（最准确但最慢）
    
    工作流程：
    1. Prefill: 首先处理 prefill_length 个 token（必须 >= residual_length）
    2. Decode: 然后逐 token 处理剩余的 token
    
    这样符合 KIVI 的预期：decode 时 value_full_length == residual_length + 1
    """
    model = model.to(device)
    if isinstance(device, str):
        device = torch.device(device)

    # 获取 KIVI 参数
    residual_length = getattr(model.config, 'residual_length', 128)
    group_size = getattr(model.config, 'group_size', 128)
    
    # 设置 prefill 长度：
    # 1. 必须 > residual_length 才能触发量化
    # 2. 必须是 group_size 的整数倍（KIVI 的 triton kernel 要求）
    if prefill_length is None:
        # 默认: residual_length + group_size，确保能触发量化
        prefill_length = residual_length + group_size
    
    if prefill_length <= residual_length:
        print(f"[WARNING] prefill_length({prefill_length}) <= residual_length({residual_length})")
        prefill_length = residual_length + group_size
    
    # 对齐到 group_size 的整数倍
    if prefill_length % group_size != 0:
        prefill_length = ((prefill_length // group_size) + 1) * group_size
        print(f"[INFO] Aligned prefill_length to group_size: {prefill_length}")
    
    if verbose:
        print("=" * 60)
        print("KIVI PPL Evaluation (Prefill + Token-by-Token Decode)")
        print("=" * 60)
        print(f"residual_length: {residual_length}")
        print(f"group_size: {group_size}")
        print(f"prefill_length: {prefill_length}")
        print(f"seqlen: {seqlen}")
        print("=" * 60)

    results = {}

    for dataset in datasets.split(","):
        dataset = dataset.strip()
        cache_testloader = (
            f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
        )
        
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
        else:
            testloader = get_ppl_eval_loaders(dataset, tokenizer, seqlen)
            torch.save(testloader, cache_testloader)
        
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        
        model.config.use_cache = True
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        loss_fct = nn.CrossEntropyLoss(reduction='sum')

        for i in tqdm(range(min(nsamples, limit)), desc=f"Eval {dataset}"):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
            seq_len = batch.size(1)
            
            if seq_len <= prefill_length:
                # 序列太短，直接一次 forward
                outputs = model(batch, use_cache=False)
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
                continue
            
            past_key_values = None
            
            # ============================================================
            # 阶段 1: Prefill - 一次性处理前 prefill_length 个 token
            # ============================================================
            prefill_tokens = batch[:, :prefill_length]
            outputs = model(
                prefill_tokens,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            
            # 计算 prefill 阶段的 loss
            prefill_logits = outputs.logits  # [batch, prefill_length, vocab_size]
            shift_logits = prefill_logits[:, :-1, :].contiguous()
            shift_labels = prefill_tokens[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
            
            # prefill 最后一个 token 预测下一个 token
            last_logits = prefill_logits[:, -1, :]
            next_target = batch[:, prefill_length]
            loss = loss_fct(last_logits.unsqueeze(1).view(-1, last_logits.size(-1)), 
                           next_target.view(-1))
            total_loss += loss.item()
            total_tokens += next_target.numel()
            
            # ============================================================
            # 阶段 2: Decode - 逐 token 处理剩余的 token
            # ============================================================
            for j in range(prefill_length, seq_len - 1):
                # 当前 token
                curr_token = batch[:, j:j+1]
                
                outputs = model(
                    curr_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                
                # 预测下一个 token
                logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
                target = batch[:, j + 1]  # [batch]
                
                loss = F.cross_entropy(logits, target, reduction='sum')
                total_loss += loss.item()
                total_tokens += target.numel()
            
            # 调试信息
            if i == 0 and verbose:
                print(f"\n[DEBUG] After sample 0:")
                print(f"  Cache type: {type(past_key_values)}")
                
                # 检查 KIVI cache 状态
                if hasattr(past_key_values, 'key_cache') and len(past_key_values.key_cache) > 0:
                    # KIVI 原版 cache 结构
                    print(f"  KIVI cache detected")
                    if hasattr(past_key_values, 'key_cache_quant'):
                        k_quant = past_key_values.key_cache_quant
                        k_full = past_key_values.key_cache_full
                        if k_quant is not None and len(k_quant) > 0 and k_quant[0] is not None:
                            print(f"  Layer 0 quantized key shape: {k_quant[0].shape}")
                        if k_full is not None and len(k_full) > 0 and k_full[0] is not None:
                            print(f"  Layer 0 full key shape: {k_full[0].shape}")
                
                elif hasattr(past_key_values, '_cache'):
                    # 自定义 KIVILatentCache 结构
                    if 0 in past_key_values._cache:
                        k_q, k_r, v_q, v_r = past_key_values._cache[0]
                        k_q_len = k_q.shape[1] if k_q is not None else 0
                        k_r_len = k_r.shape[1] if k_r is not None else 0
                        print(f"  Quantized tokens: {k_q_len}")
                        print(f"  Residual tokens: {k_r_len}")
                        print(f"  Total: {k_q_len + k_r_len}")
                
                elif isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                    print(f"  Tuple cache, layer 0 key shape: {past_key_values[0][0].shape}")

        ppl = torch.exp(torch.tensor(total_loss / total_tokens))
        results[dataset] = ppl.item()
        
        if verbose:
            print(f"\n{dataset} PPL: {ppl.item():.4f}")
            print(f"Total tokens evaluated: {total_tokens}")

    return results


@torch.no_grad()
def compare_ppl_with_without_kivi(
    model_kivi,
    model_baseline,
    tokenizer,
    model_name: str,
    datasets: str = "wikitext2",
    seqlen: int = 2048,
    device: str = "cuda",
    limit: int = 100,
):
    """
    对比 KIVI 量化和无量化的 PPL
    """
    print("=" * 60)
    print("Comparing KIVI vs Baseline PPL")
    print("=" * 60)
    
    # KIVI 模型
    print("\n[1/2] Evaluating KIVI model...")
    results_kivi = eval_ppl_kivi(
        model_kivi, tokenizer, model_name + "_kivi",
        datasets, seqlen, device, limit,
        chunk_size=256, verbose=True
    )
    
    # Baseline 模型（无量化）
    print("\n[2/2] Evaluating Baseline model...")
    results_baseline = eval_ppl_kivi(
        model_baseline, tokenizer, model_name + "_baseline",
        datasets, seqlen, device, limit,
        chunk_size=256, verbose=True
    )
    
    # 对比结果
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Baseline':<15} {'KIVI':<15} {'Δ PPL':<15}")
    print("-" * 60)
    
    for dataset in results_kivi:
        baseline_ppl = results_baseline.get(dataset, float('nan'))
        kivi_ppl = results_kivi[dataset]
        delta = kivi_ppl - baseline_ppl
        print(f"{dataset:<15} {baseline_ppl:<15.4f} {kivi_ppl:<15.4f} {delta:+.4f}")
    
    print("=" * 60)
    
    return {"kivi": results_kivi, "baseline": results_baseline}


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # 示例用法
    print("""
使用示例:

1. 基本用法:
    
    from eval_ppl_kivi import eval_ppl_kivi
    
    results = eval_ppl_kivi(
        model=model,
        tokenizer=tokenizer,
        model_name="llama3.1-8b",
        datasets="wikitext2",
        seqlen=2048,
        chunk_size=256,  # 必须 > residual_length
        limit=100,
    )
    print(f"PPL: {results}")

2. 逐 token 模式（更准确但更慢）:
    
    from eval_ppl_kivi import eval_ppl_kivi_token_by_token
    
    results = eval_ppl_kivi_token_by_token(
        model=model,
        tokenizer=tokenizer,
        model_name="llama3.1-8b",
        datasets="wikitext2",
        seqlen=2048,
        limit=50,  # 建议较小的 limit
    )

3. 对比 KIVI vs Baseline:
    
    from eval_ppl_kivi import compare_ppl_with_without_kivi
    
    results = compare_ppl_with_without_kivi(
        model_kivi=model_kivi,
        model_baseline=model_baseline,
        tokenizer=tokenizer,
        model_name="llama3.1-8b",
        datasets="wikitext2",
    )
""")
