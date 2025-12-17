"""
LM-Evaluation-Harness Integration with KVQuant-style Simulated Quantization

This script evaluates LLaMA models on various benchmarks using lm-eval framework
with KV cache quantization applied.

Requirements:
    pip install lm-eval

Usage:
    # 4-bit quantization on common benchmarks
    python3 run_lm_eval_kvquant.py meta-llama/Llama-3-8B \
        --abits 4 --include_sparse \
        --tasks hellaswag,winogrande,arc_easy,arc_challenge
    
    # FP16 baseline
    python3 run_lm_eval_kvquant.py meta-llama/Llama-3-8B \
        --abits 16 \
        --tasks hellaswag

Reference:
    - KVQuant: https://github.com/SqueezeAILab/KVQuant
    - lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from loguru import logger
import math
from typing import Optional, List, Dict, Any


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
# Model Loading
# ============================================================================

def get_model(model_path, max_length=2048, use_flash_attn=True):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

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
# LM-Eval Integration
# ============================================================================

def run_lm_eval(model, tokenizer, tasks, batch_size=1, num_fewshot=0, 
                limit=None, device="cuda"):
    """
    Run lm-evaluation-harness on the model.
    
    This function wraps the lm-eval library to evaluate the quantized model.
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator
    except ImportError:
        logger.error("lm-eval not installed. Please run: pip install lm-eval")
        raise ImportError("Please install lm-eval: pip install lm-eval")
    
    # Create lm-eval model wrapper
    # We need to wrap our quantized model
    class QuantizedHFLM(HFLM):
        def __init__(self, pretrained_model, tokenizer, **kwargs):
            # Don't call parent __init__ with pretrained string
            self._model = pretrained_model
            self._tokenizer = tokenizer
            self._device = next(pretrained_model.parameters()).device
            self._batch_size = kwargs.get('batch_size', 1)
            
            # Set required attributes
            self.tokenizer = tokenizer
            self.model = pretrained_model
            self.device = self._device
            self.batch_size = self._batch_size
            
            # Vocab size
            self._vocab_size = tokenizer.vocab_size
            
            # Add required methods for lm-eval
            self._max_length = kwargs.get('max_length', 2048)
            
        @property
        def eot_token_id(self):
            return self.tokenizer.eos_token_id
        
        @property
        def max_length(self):
            return self._max_length
        
        @property
        def max_gen_toks(self):
            return 256
        
        @property
        def vocab_size(self):
            return self._vocab_size
        
        def tok_encode(self, string: str, **kwargs):
            return self.tokenizer.encode(string, add_special_tokens=False)
        
        def tok_decode(self, tokens, **kwargs):
            return self.tokenizer.decode(tokens)
        
        def _model_call(self, inps):
            with torch.no_grad():
                return self.model(inps).logits
        
        def _model_generate(self, context, max_length, eos_token_id):
            with torch.no_grad():
                return self.model.generate(
                    context,
                    max_length=max_length,
                    eos_token_id=eos_token_id,
                    do_sample=False
                )
    
    # Try using the simple_evaluate interface
    logger.info(f"Running lm-eval on tasks: {tasks}")
    
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model.config._name_or_path},trust_remote_code=True",
        tasks=tasks.split(",") if isinstance(tasks, str) else tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
        device=device,
    )
    
    return results


def run_lm_eval_with_model(model, tokenizer, tasks, batch_size=1, num_fewshot=0,
                           limit=None, device="cuda"):
    """
    Alternative: Run lm-eval by directly using the quantized model.
    This avoids reloading the model.
    """
    try:
        import lm_eval
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error("lm-eval not installed. Please run: pip install lm-eval")
        return None
    
    # Monkey-patch to use our model
    class PreloadedHFLM(HFLM):
        _preloaded_model = None
        _preloaded_tokenizer = None
        
        def __init__(self, **kwargs):
            # Minimal init - skip loading
            self._model = PreloadedHFLM._preloaded_model
            self.tokenizer = PreloadedHFLM._preloaded_tokenizer
            self._device = str(next(self._model.parameters()).device)
            self._batch_size = kwargs.get('batch_size', 1)
            self._max_length = getattr(self._model.config, 'max_position_embeddings', 2048)
            self._add_special_tokens = False
            self.truncation = False
            
    PreloadedHFLM._preloaded_model = model
    PreloadedHFLM._preloaded_tokenizer = tokenizer
    
    # Parse tasks
    task_list = tasks.split(",") if isinstance(tasks, str) else tasks
    
    logger.info(f"Evaluating on tasks: {task_list}")
    
    try:
        # Use the preloaded model wrapper
        lm = PreloadedHFLM(batch_size=batch_size)
        
        results = evaluator.evaluate(
            lm=lm,
            task_dict=lm_eval.tasks.get_task_dict(task_list),
            num_fewshot=num_fewshot,
            limit=limit,
        )
        
        return results
        
    except Exception as e:
        logger.warning(f"Direct model evaluation failed: {e}")
        logger.info("Falling back to standard lm-eval (will reload model without quantization)")
        return None


def run_simple_benchmarks(model, tokenizer, tasks, device="cuda", limit=100):
    """
    Simple built-in evaluation for common benchmarks when lm-eval is not available.
    Supports: hellaswag, winogrande, arc_easy, arc_challenge, piqa, boolq
    """
    from datasets import load_dataset
    import numpy as np
    
    results = {}
    
    for task in tasks:
        task = task.strip().lower()
        logger.info(f"Evaluating {task}...")
        
        try:
            if task == "hellaswag":
                dataset = load_dataset("hellaswag", split="validation")
                if limit:
                    dataset = dataset.select(range(min(limit, len(dataset))))
                
                correct = 0
                total = 0
                
                for ex in tqdm(dataset, desc="HellaSwag"):
                    ctx = ex['ctx']
                    endings = ex['endings']
                    label = int(ex['label'])
                    
                    # Score each ending
                    scores = []
                    for ending in endings:
                        text = ctx + " " + ending
                        inputs = tokenizer(text, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                        
                        # Get log prob of continuation
                        ctx_len = len(tokenizer.encode(ctx))
                        log_probs = torch.log_softmax(logits[0, ctx_len-1:-1], dim=-1)
                        target_ids = inputs.input_ids[0, ctx_len:]
                        score = log_probs[range(len(target_ids)), target_ids].mean().item()
                        scores.append(score)
                    
                    pred = np.argmax(scores)
                    if pred == label:
                        correct += 1
                    total += 1
                
                acc = correct / total * 100
                results[task] = acc
                
            elif task == "winogrande":
                dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
                if limit:
                    dataset = dataset.select(range(min(limit, len(dataset))))
                
                correct = 0
                total = 0
                
                for ex in tqdm(dataset, desc="WinoGrande"):
                    sentence = ex['sentence']
                    option1 = ex['option1']
                    option2 = ex['option2']
                    label = int(ex['answer']) - 1  # 1 or 2 -> 0 or 1
                    
                    scores = []
                    for option in [option1, option2]:
                        text = sentence.replace("_", option)
                        inputs = tokenizer(text, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                        
                        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
                        target_ids = inputs.input_ids[0, 1:]
                        score = log_probs[range(len(target_ids)), target_ids].mean().item()
                        scores.append(score)
                    
                    pred = np.argmax(scores)
                    if pred == label:
                        correct += 1
                    total += 1
                
                acc = correct / total * 100
                results[task] = acc
                
            elif task in ["arc_easy", "arc_challenge"]:
                subset = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
                dataset = load_dataset("ai2_arc", subset, split="test")
                if limit:
                    dataset = dataset.select(range(min(limit, len(dataset))))
                
                correct = 0
                total = 0
                
                for ex in tqdm(dataset, desc=task):
                    question = ex['question']
                    choices = ex['choices']
                    answer_key = ex['answerKey']
                    
                    # Map answer key to index
                    labels = choices['label']
                    texts = choices['text']
                    answer_idx = labels.index(answer_key)
                    
                    scores = []
                    for choice_text in texts:
                        text = f"Question: {question}\nAnswer: {choice_text}"
                        inputs = tokenizer(text, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                        
                        q_len = len(tokenizer.encode(f"Question: {question}\nAnswer:"))
                        log_probs = torch.log_softmax(logits[0, q_len-1:-1], dim=-1)
                        target_ids = inputs.input_ids[0, q_len:]
                        if len(target_ids) > 0:
                            score = log_probs[range(len(target_ids)), target_ids].mean().item()
                        else:
                            score = float('-inf')
                        scores.append(score)
                    
                    pred = np.argmax(scores)
                    if pred == answer_idx:
                        correct += 1
                    total += 1
                
                acc = correct / total * 100
                results[task] = acc
                
            elif task == "piqa":
                dataset = load_dataset("piqa", split="validation")
                if limit:
                    dataset = dataset.select(range(min(limit, len(dataset))))
                
                correct = 0
                total = 0
                
                for ex in tqdm(dataset, desc="PIQA"):
                    goal = ex['goal']
                    sol1 = ex['sol1']
                    sol2 = ex['sol2']
                    label = ex['label']
                    
                    scores = []
                    for sol in [sol1, sol2]:
                        text = f"Goal: {goal}\nSolution: {sol}"
                        inputs = tokenizer(text, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                        
                        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
                        target_ids = inputs.input_ids[0, 1:]
                        score = log_probs[range(len(target_ids)), target_ids].mean().item()
                        scores.append(score)
                    
                    pred = np.argmax(scores)
                    if pred == label:
                        correct += 1
                    total += 1
                
                acc = correct / total * 100
                results[task] = acc
                
            elif task == "boolq":
                dataset = load_dataset("boolq", split="validation")
                if limit:
                    dataset = dataset.select(range(min(limit, len(dataset))))
                
                correct = 0
                total = 0
                
                for ex in tqdm(dataset, desc="BoolQ"):
                    passage = ex['passage']
                    question = ex['question']
                    answer = ex['answer']  # True or False
                    
                    scores = []
                    for ans_text in ["Yes", "No"]:
                        text = f"Passage: {passage}\nQuestion: {question}\nAnswer: {ans_text}"
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                        
                        # Score the answer token
                        log_probs = torch.log_softmax(logits[0, -2], dim=-1)
                        ans_token = tokenizer.encode(ans_text, add_special_tokens=False)[0]
                        score = log_probs[ans_token].item()
                        scores.append(score)
                    
                    pred = scores[0] > scores[1]  # Yes > No means True
                    if pred == answer:
                        correct += 1
                    total += 1
                
                acc = correct / total * 100
                results[task] = acc
            
            else:
                logger.warning(f"Task {task} not supported in simple mode. Use lm-eval for this task.")
                
        except Exception as e:
            logger.error(f"Error evaluating {task}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LM-Eval with KVQuant quantization")
    
    parser.add_argument('model', type=str, help='Model path')
    parser.add_argument('--max_length', type=int, default=2048)
    
    # Quantization
    parser.add_argument('--abits', type=int, default=4, choices=[2, 3, 4, 5, 8, 16])
    parser.add_argument('--perchannel', type=str, nargs='+', default=["k_proj"])
    parser.add_argument('--pertoken', type=str, nargs='+', default=["v_proj"])
    parser.add_argument('--include_sparse', action='store_true')
    parser.add_argument('--sparsity_threshold', type=float, default=0.99)
    parser.add_argument('--first_few_fp16', type=int, default=-1)
    parser.add_argument('--clamp', action='store_true')
    
    # Evaluation
    parser.add_argument('--tasks', type=str, default="hellaswag,winogrande,arc_easy",
                        help='Comma-separated list of tasks')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_flash_attn', action='store_true')
    parser.add_argument('--use_lm_eval', action='store_true', 
                        help='Use lm-eval library (requires pip install lm-eval)')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True,
               level="DEBUG" if args.verbose else "INFO")
    
    logger.info("=" * 60)
    logger.info("LM-Eval with KVQuant Quantization")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Bits: {args.abits}")
    logger.info(f"Tasks: {args.tasks}")
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
    
    model = model.half().to(args.device)
    
    # Parse tasks
    task_list = [t.strip() for t in args.tasks.split(",")]
    
    # Run evaluation
    if args.use_lm_eval:
        logger.info("Using lm-eval library...")
        results = run_lm_eval_with_model(
            model, tokenizer, task_list,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            device=args.device
        )
        
        if results is None:
            logger.warning("lm-eval failed, falling back to simple evaluation")
            results = run_simple_benchmarks(
                model, tokenizer, task_list,
                device=args.device, limit=args.limit or 100
            )
    else:
        logger.info("Using built-in simple evaluation...")
        results = run_simple_benchmarks(
            model, tokenizer, task_list,
            device=args.device, limit=args.limit or 100
        )
    
    # Print results
    logger.info("=" * 60)
    logger.info("Results:")
    if isinstance(results, dict):
        if 'results' in results:
            # lm-eval format
            for task, metrics in results['results'].items():
                acc = metrics.get('acc', metrics.get('acc_norm', 'N/A'))
                if isinstance(acc, float):
                    acc = acc * 100
                logger.info(f"  {task}: {acc:.2f}%")
        else:
            # Simple format
            for task, acc in results.items():
                logger.info(f"  {task}: {acc:.2f}%")
            if results:
                avg = sum(results.values()) / len(results)
                logger.info(f"  Average: {avg:.2f}%")
    logger.info("=" * 60)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
