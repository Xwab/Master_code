"""
LongBench Evaluation with KVQuant-style Simulated Quantization

This script evaluates LLaMA models on LongBench benchmark with KV cache quantization.
LongBench tests long-context understanding across various tasks.

Usage:
    # 4-bit quantization
    python3 run_longbench_kvquant.py meta-llama/Llama-3-8B --abits 4 --include_sparse
    
    # FP16 baseline
    python3 run_longbench_kvquant.py meta-llama/Llama-3-8B --abits 16

Reference: 
    - KVQuant: https://github.com/SqueezeAILab/KVQuant
    - LongBench: https://github.com/THUDM/LongBench
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention,
    apply_rotary_pos_emb, repeat_kv
)
from loguru import logger
import math
from typing import Optional, Tuple, List, Dict
import numpy as np


# ============================================================================
# KVQuant Quantization Functions (same as kvquant_simquant_ppl_eval.py)
# ============================================================================

def get_outliers_dynamic(w, channel=-1, thresh=0.999, first_few_fp16=-1):
    t = 1 - ((1 - thresh) / 2)
    w = w.float()
    
    outlier_threshold_upper = torch.quantile(w, t, dim=channel)
    outlier_threshold_lower = torch.quantile(w, 1 - t, dim=channel)
    
    outlier_threshold_upper = outlier_threshold_upper.unsqueeze(channel)
    outlier_threshold_lower = outlier_threshold_lower.unsqueeze(channel)
    
    under_lower = w <= outlier_threshold_lower
    above_upper = w >= outlier_threshold_upper
    
    outlier_mask = torch.logical_or(under_lower, above_upper)
    
    if first_few_fp16 > -1:
        outlier_mask[:first_few_fp16, :] = True
    
    return outlier_mask


def quant_fn_zp(inp, bits=8, qchannel=-1, dynamicquantization=True,
                include_sparse=False, outlier_mask=None, clamp=False):
    if dynamicquantization:
        if include_sparse and outlier_mask is not None:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values.unsqueeze(qchannel)
            median_mask = median * outlier_mask
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values
    
    rangeval = (maxval - minval)
    qx = (2**bits - 1) / rangeval
    
    if clamp:
        offset = torch.round(minval * qx).clamp(-(2**bits - 1), 0)
    else:
        offset = minval * qx
    
    offset = offset.unsqueeze(qchannel)
    qx = qx.unsqueeze(qchannel)
    
    if include_sparse and outlier_mask is not None:
        outliers = inp * outlier_mask
        inp = inp - outliers
    
    qinp = torch.round(qx * inp - offset)
    qinp = torch.clip(qinp, min=0, max=2**bits - 1)
    qinp_out = (qinp + offset) / qx
    
    if include_sparse and outlier_mask is not None:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers
    
    return torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)


class QuantLinearSim(nn.Module):
    def __init__(self, name, bits, infeatures, outfeatures, weight, bias,
                 perchannel=True, include_sparse=False, sparsity_threshold=0.999,
                 first_few_fp16=-1, clamp=False):
        super().__init__()
        self.name = name
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.weight = weight.T.detach().clone()
        self.bias = bias.detach().clone() if bias is not None else None
        self.perchannel = perchannel
        self.qchannel = 0 if perchannel else -1
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.first_few_fp16 = first_few_fp16
        self.clamp = clamp
    
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        
        self.weight = self.weight.to(x.device)
        if self.bias is not None:
            self.bias = self.bias.to(x.device)
        
        x = x.half()
        y = x @ self.weight
        y = y + self.bias if self.bias is not None else y
        y = y.float()
        
        if self.bits >= 16:
            return y.reshape(out_shape).half()
        
        if self.include_sparse:
            outlier_mask = get_outliers_dynamic(
                y, channel=self.qchannel, thresh=self.sparsity_threshold,
                first_few_fp16=self.first_few_fp16
            )
        else:
            outlier_mask = None
        
        y = quant_fn_zp(
            y, bits=self.bits, qchannel=self.qchannel,
            include_sparse=self.include_sparse, outlier_mask=outlier_mask,
            dynamicquantization=True, clamp=self.clamp
        )
        
        return y.reshape(out_shape).half()


def make_quant_sim(model, bits, perchannel_match=["k_proj"], pertoken_match=["v_proj"],
                   include_sparse=False, sparsity_threshold=0.999, first_few_fp16=-1, clamp=False):
    replaced = 0
    for name, module in list(model.named_modules()):
        is_perchannel = any(p in name for p in perchannel_match)
        is_pertoken = any(p in name for p in pertoken_match)
        
        if (is_perchannel or is_pertoken) and isinstance(module, nn.Linear):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            
            quant_layer = QuantLinearSim(
                name=name, bits=bits, infeatures=module.in_features,
                outfeatures=module.out_features, weight=module.weight, bias=module.bias,
                perchannel=is_perchannel, include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold, first_few_fp16=first_few_fp16, clamp=clamp
            )
            setattr(parent, attr_name, quant_layer)
            replaced += 1
    
    logger.info(f"Replaced {replaced} layers with QuantLinearSim")
    return model


# ============================================================================
# LongBench Evaluation
# ============================================================================

LONGBENCH_DATASETS = [
    # Single-Document QA
    "narrativeqa", "qasper", "multifieldqa_en",
    # Multi-Document QA  
    "hotpotqa", "2wikimqa", "musique",
    # Summarization
    "gov_report", "qmsum", "multi_news",
    # Few-shot Learning
    "trec", "triviaqa", "samsum",
    # Synthetic Tasks
    "passage_count", "passage_retrieval_en",
    # Code
    "lcc", "repobench-p"
]

LONGBENCH_MAXLEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32,
    "lcc": 64, "repobench-p": 64
}


def build_longbench_prompt(example, dataset_name):
    """Build prompt for LongBench task."""
    if dataset_name in ["narrativeqa", "qasper", "multifieldqa_en", 
                        "hotpotqa", "2wikimqa", "musique"]:
        # QA format
        prompt = f"Read the following text and answer the question.\n\n"
        prompt += f"Text: {example['context']}\n\n"
        prompt += f"Question: {example['input']}\n\n"
        prompt += f"Answer:"
    elif dataset_name in ["gov_report", "qmsum", "multi_news"]:
        # Summarization format
        prompt = f"Summarize the following text.\n\n"
        prompt += f"Text: {example['context']}\n\n"
        prompt += f"Summary:"
    elif dataset_name == "trec":
        prompt = f"Classify the following question.\n\n"
        prompt += f"Question: {example['input']}\n\n"
        prompt += f"Category:"
    elif dataset_name == "triviaqa":
        prompt = f"Answer the following question based on the context.\n\n"
        prompt += f"Context: {example['context']}\n\n"
        prompt += f"Question: {example['input']}\n\n"
        prompt += f"Answer:"
    elif dataset_name == "samsum":
        prompt = f"Summarize the following dialogue.\n\n"
        prompt += f"Dialogue: {example['context']}\n\n"
        prompt += f"Summary:"
    elif dataset_name in ["passage_count", "passage_retrieval_en"]:
        prompt = f"{example['context']}\n\n{example['input']}"
    elif dataset_name in ["lcc", "repobench-p"]:
        prompt = f"Complete the following code.\n\n{example['context']}\n{example['input']}"
    else:
        prompt = f"{example['context']}\n\n{example['input']}"
    
    return prompt


def compute_score(prediction: str, answers: List[str], dataset_name: str) -> float:
    """Compute score for a single prediction."""
    prediction = prediction.strip().lower()
    
    if dataset_name in ["narrativeqa", "qasper", "multifieldqa_en", 
                        "hotpotqa", "2wikimqa", "musique", "triviaqa"]:
        # F1 score for QA
        def f1_score(pred, answer):
            pred_tokens = pred.split()
            answer_tokens = answer.lower().split()
            common = set(pred_tokens) & set(answer_tokens)
            if len(common) == 0:
                return 0
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(answer_tokens)
            return 2 * precision * recall / (precision + recall)
        
        return max(f1_score(prediction, ans) for ans in answers)
    
    elif dataset_name in ["gov_report", "qmsum", "multi_news", "samsum"]:
        # ROUGE-L for summarization (simplified)
        def rouge_l(pred, ref):
            pred_tokens = pred.split()
            ref_tokens = ref.lower().split()
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                return 0
            
            # LCS
            m, n = len(pred_tokens), len(ref_tokens)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if pred_tokens[i-1] == ref_tokens[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            lcs = dp[m][n]
            
            precision = lcs / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0
            if precision + recall == 0:
                return 0
            return 2 * precision * recall / (precision + recall)
        
        return max(rouge_l(prediction, ans) for ans in answers)
    
    elif dataset_name == "trec":
        # Accuracy for classification
        for ans in answers:
            if ans.lower() in prediction:
                return 1.0
        return 0.0
    
    elif dataset_name in ["passage_count", "passage_retrieval_en"]:
        # Exact match
        for ans in answers:
            if ans.lower() == prediction:
                return 1.0
        return 0.0
    
    elif dataset_name in ["lcc", "repobench-p"]:
        # Edit similarity for code
        def edit_sim(pred, ref):
            m, n = len(pred), len(ref)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if pred[i-1] == ref[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            return 1 - dp[m][n] / max(m, n) if max(m, n) > 0 else 0
        
        return max(edit_sim(prediction, ans) for ans in answers)
    
    else:
        # Default: exact match
        for ans in answers:
            if ans.lower() in prediction:
                return 1.0
        return 0.0


@torch.no_grad()
def evaluate_longbench(model, tokenizer, dataset_names, max_length=4096, 
                       device="cuda", limit=None):
    """Evaluate model on LongBench datasets."""
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for dataset_name in dataset_names:
        logger.info(f"Evaluating {dataset_name}...")
        
        try:
            dataset = load_dataset("THUDM/LongBench", dataset_name, split="test")
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            continue
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        max_gen_len = LONGBENCH_MAXLEN.get(dataset_name, 64)
        scores = []
        
        for example in tqdm(dataset, desc=dataset_name):
            prompt = build_longbench_prompt(example, dataset_name)
            
            # Tokenize
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length - max_gen_len
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode
            generated = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Get answers
            answers = example.get('answers', [example.get('answer', '')])
            if isinstance(answers, str):
                answers = [answers]
            
            # Score
            score = compute_score(generated, answers, dataset_name)
            scores.append(score)
        
        avg_score = np.mean(scores) * 100
        results[dataset_name] = avg_score
        logger.info(f"  {dataset_name}: {avg_score:.2f}")
    
    return results


# ============================================================================
# Model Loading
# ============================================================================

def get_model(model_path, max_length=4096, use_flash_attn=True):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # RoPE scaling for long sequences
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        logger.info(f"Applied RoPE scaling: {scaling_factor}")

    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True,
            attn_implementation=attn_impl, torch_dtype=torch.float16
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True, torch_dtype=torch.float16
        )
    
    model.eval()
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LongBench evaluation with KVQuant")
    
    parser.add_argument('model', type=str, help='Model path')
    parser.add_argument('--max_length', type=int, default=4096)
    
    # Quantization
    parser.add_argument('--abits', type=int, default=4, choices=[2, 3, 4, 5, 8, 16])
    parser.add_argument('--perchannel', type=str, nargs='+', default=["k_proj"])
    parser.add_argument('--pertoken', type=str, nargs='+', default=["v_proj"])
    parser.add_argument('--include_sparse', action='store_true')
    parser.add_argument('--sparsity_threshold', type=float, default=0.99)
    parser.add_argument('--first_few_fp16', type=int, default=-1)
    parser.add_argument('--clamp', action='store_true')
    
    # Evaluation
    parser.add_argument('--datasets', type=str, nargs='+', default=["narrativeqa", "qasper", "hotpotqa"])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_flash_attn', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True,
               level="DEBUG" if args.verbose else "INFO")
    
    logger.info("=" * 60)
    logger.info("LongBench Evaluation with KVQuant")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Bits: {args.abits}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Max length: {args.max_length}")
    if args.include_sparse:
        logger.info(f"Sparse: {(1-args.sparsity_threshold)*100:.1f}% outliers in FP16")
    logger.info("=" * 60)
    
    # Load model
    logger.info("Loading model...")
    model = get_model(args.model, args.max_length, not args.no_flash_attn)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply quantization
    if args.abits < 16:
        logger.info(f"Applying {args.abits}-bit KVQuant quantization...")
        model = make_quant_sim(
            model, args.abits, args.perchannel, args.pertoken,
            args.include_sparse, args.sparsity_threshold,
            args.first_few_fp16, args.clamp
        )
    
    model = model.half()
    
    # Evaluate
    results = evaluate_longbench(
        model, tokenizer, args.datasets, 
        args.max_length, args.device, args.limit
    )
    
    # Print results
    logger.info("=" * 60)
    logger.info("Results:")
    for dataset, score in results.items():
        logger.info(f"  {dataset}: {score:.2f}")
    if results:
        logger.info(f"  Average: {np.mean(list(results.values())):.2f}")
    logger.info("=" * 60)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
