"""
PPL Evaluation for Qwen with KIVI Quantized KV Cache

This script evaluates perplexity on WikiText-2 and other datasets,
comparing standard cache vs KIVI quantized cache.

Usage:
    python eval_ppl_qwen_kivi.py --model Qwen/Qwen2-7B-Instruct --k_bits 2 --v_bits 2
"""

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from modules.kivi_cache_general import KIVICache


def load_wikitext2(tokenizer, max_length: int = 2048, split: str = "test"):
    """Load WikiText-2 dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    # Concatenate all text
    text = "\n\n".join(dataset["text"])
    
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    
    return input_ids


def eval_ppl_standard(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """
    Evaluate PPL with standard (DynamicCache) cache.
    Uses sliding window for long sequences.
    """
    device = next(model.parameters()).device
    seq_len = input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Standard PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        target_chunk = input_chunk.clone()
        target_chunk[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk, use_cache=False)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood.item())
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break
    
    ppl = math.exp(sum(nlls) / prev_end_loc)
    return ppl


def eval_ppl_kivi(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """
    Evaluate PPL with KIVI quantized cache.
    Uses sliding window for long sequences.
    """
    device = next(model.parameters()).device
    seq_len = input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc=f"KIVI-{k_bits}bit PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        target_chunk = input_chunk.clone()
        target_chunk[:, :-trg_len] = -100
        
        # Create fresh KIVI cache for each chunk
        cache = KIVICache(
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
        )
        
        with torch.no_grad():
            outputs = model(
                input_chunk,
                labels=target_chunk,
                past_key_values=cache,
                use_cache=True,
            )
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood.item())
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break
    
    ppl = math.exp(sum(nlls) / prev_end_loc)
    return ppl


def eval_ppl_kivi_incremental(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
    chunk_size: int = 256,
    max_tokens: int = 4096,
) -> float:
    """
    Evaluate PPL with KIVI cache using incremental (prefill + decode) approach.
    This better simulates real inference where cache is reused.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        input_ids: Input token ids [1, seq_len]
        k_bits, v_bits: Quantization bits
        group_size: Quantization group size
        residual_length: Full precision residual length
        chunk_size: Number of tokens to process at a time
        max_tokens: Maximum tokens to evaluate
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    seq_len = min(input_ids.size(1), max_tokens)
    
    # Align chunk_size to group_size for proper quantization
    if group_size > 0:
        chunk_size = max(chunk_size, group_size)
        chunk_size = (chunk_size // group_size) * group_size
    
    # Create KIVI cache
    cache = KIVICache(
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
    )
    
    total_loss = 0.0
    total_tokens = 0
    
    # Process in chunks
    for start_idx in tqdm(range(0, seq_len, chunk_size), desc=f"KIVI-{k_bits}bit Incremental"):
        end_idx = min(start_idx + chunk_size, seq_len)
        
        chunk_ids = input_ids[:, start_idx:end_idx]
        
        with torch.no_grad():
            outputs = model(
                input_ids=chunk_ids,
                past_key_values=cache,
                use_cache=True,
            )
        
        # Calculate loss for this chunk (except first token which has no previous context)
        if start_idx > 0:
            # All tokens in this chunk can be evaluated
            logits = outputs.logits[:, :-1, :]  # [B, chunk_len-1, vocab]
            targets = chunk_ids[:, 1:]  # [B, chunk_len-1]
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
        else:
            # First chunk: evaluate from second token onwards
            logits = outputs.logits[:, :-1, :]
            targets = chunk_ids[:, 1:]
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
    
    # Print cache stats
    print(f"\nCache stats: {cache.get_cache_info()}")
    
    return ppl


def run_evaluation(
    model_path: str,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
    max_length: int = 2048,
    stride: int = 512,
    max_tokens: int = 4096,
    use_incremental: bool = False,
):
    """Run full evaluation."""
    print("=" * 70)
    print("KIVI PPL Evaluation")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"KIVI config: k_bits={k_bits}, v_bits={v_bits}, "
          f"group_size={group_size}, residual={residual_length}")
    print(f"Eval config: max_length={max_length}, stride={stride}")
    
    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    
    # Load dataset
    print("\nLoading WikiText-2...")
    input_ids = load_wikitext2(tokenizer)
    print(f"Total tokens: {input_ids.size(1)}")
    
    # Truncate for faster evaluation
    if input_ids.size(1) > max_tokens:
        input_ids = input_ids[:, :max_tokens]
        print(f"Truncated to: {input_ids.size(1)} tokens")
    
    results = {}
    
    # Standard cache evaluation
    print("\n" + "-" * 50)
    print("Evaluating with Standard Cache (FP16)...")
    ppl_std = eval_ppl_standard(model, tokenizer, input_ids, max_length, stride)
    results["standard_fp16"] = ppl_std
    print(f"Standard PPL: {ppl_std:.4f}")
    
    # KIVI evaluation
    print("\n" + "-" * 50)
    print(f"Evaluating with KIVI Cache ({k_bits}-bit K, {v_bits}-bit V)...")
    
    if use_incremental:
        ppl_kivi = eval_ppl_kivi_incremental(
            model, tokenizer, input_ids,
            k_bits=k_bits, v_bits=v_bits,
            group_size=group_size, residual_length=residual_length,
            max_tokens=max_tokens,
        )
    else:
        ppl_kivi = eval_ppl_kivi(
            model, tokenizer, input_ids,
            k_bits=k_bits, v_bits=v_bits,
            group_size=group_size, residual_length=residual_length,
            max_length=max_length, stride=stride,
        )
    results[f"kivi_{k_bits}bit"] = ppl_kivi
    print(f"KIVI PPL: {ppl_kivi:.4f}")
    
    # Additional bit configurations
    for bits in [4, 8]:
        if bits != k_bits:
            print(f"\nEvaluating KIVI {bits}-bit...")
            if use_incremental:
                ppl = eval_ppl_kivi_incremental(
                    model, tokenizer, input_ids,
                    k_bits=bits, v_bits=bits,
                    group_size=group_size, residual_length=residual_length,
                    max_tokens=max_tokens,
                )
            else:
                ppl = eval_ppl_kivi(
                    model, tokenizer, input_ids,
                    k_bits=bits, v_bits=bits,
                    group_size=group_size, residual_length=residual_length,
                    max_length=max_length, stride=stride,
                )
            results[f"kivi_{bits}bit"] = ppl
            print(f"KIVI-{bits}bit PPL: {ppl:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Configuration':<25} {'PPL':>10} {'Δ PPL':>10} {'Compression':>15}")
    print("-" * 70)
    
    base_ppl = results["standard_fp16"]
    for name, ppl in sorted(results.items()):
        delta = ppl - base_ppl
        
        if "kivi" in name:
            bits = int(name.split("_")[1].replace("bit", ""))
            # Approximate compression (assuming long sequence)
            compression = f"~{16 / bits:.1f}x"
        else:
            compression = "1x (baseline)"
        
        print(f"{name:<25} {ppl:>10.4f} {delta:>+10.4f} {compression:>15}")
    
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="KIVI PPL Evaluation for Qwen")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B",
                        help="Model name or path")
    parser.add_argument("--k_bits", type=int, default=2,
                        help="Key quantization bits")
    parser.add_argument("--v_bits", type=int, default=2,
                        help="Value quantization bits")
    parser.add_argument("--group_size", type=int, default=32,
                        help="Quantization group size")
    parser.add_argument("--residual_length", type=int, default=128,
                        help="Full precision residual length")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max sequence length per chunk")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride for sliding window")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max total tokens to evaluate")
    parser.add_argument("--incremental", action="store_true",
                        help="Use incremental evaluation (simulates real inference)")
    
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
        residual_length=args.residual_length,
        max_length=args.max_length,
        stride=args.stride,
        max_tokens=args.max_tokens,
        use_incremental=args.incremental,
    )


if __name__ == "__main__":
    main()
