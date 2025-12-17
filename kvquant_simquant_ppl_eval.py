"""
KVQuant-style Simulated Quantization for LLaMA3 with PPL Evaluation

This script combines KVQuant's simulated quantization approach with custom PPL evaluation.
It supports GQA (Grouped Query Attention) architecture used in LLaMA3.

Reference: https://github.com/SqueezeAILab/KVQuant
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from loguru import logger
import math
from typing import Optional, Tuple
import pickle


# ============================================================================
# Quantization Functions (from KVQuant simquant_module_quantizer.py)
# ============================================================================

def round_to_nearest_pole_sim(w, poles):
    """
    Round the numbers in w to the nearest value in poles (for NUQ).
    """
    stack = []
    for c in poles:
        diff = (w - c).abs()
        stack.append(diff)
    diff = torch.stack(stack)
    idx = diff.argmin(axis=0)
    aug = torch.zeros_like(w)
    for i, c in enumerate(poles):
        aug += (idx == i) * c
    return aug


def get_outliers_dynamic(w, channel=-1, thresh=0.999, first_few_fp16=-1):
    """
    Dynamically detect outliers above/below threshold percentiles.
    """
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


def quant_fn_zp(
    inp,
    bits=8,
    qchannel=-1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    clamp=False
):
    """
    Performs simulated integer quantization with zero-point.
    """
    # Set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse and outlier_mask is not None:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values

    # Compute scale and offset
    rangeval = (maxval - minval)
    qx = (2**bits - 1) / rangeval.clamp(min=1e-8)

    if clamp:
        offset = torch.round(minval * qx)
        offset = offset.clamp(-(2**bits - 1), 0)
    else:
        offset = minval * qx

    offset = offset.unsqueeze(qchannel)
    qx = qx.unsqueeze(qchannel)

    # Handle outlier removal
    if include_sparse and outlier_mask is not None:
        outliers = inp * outlier_mask
        inp = inp - outliers

    # Quantize and dequantize
    qinp = torch.round(qx * inp - offset)
    qinp = torch.clip(qinp, min=0, max=2**bits - 1)
    qinp_out = (qinp + offset) / qx

    # Add outliers back
    if include_sparse and outlier_mask is not None:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers

    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)
    return qinp_out


# ============================================================================
# Simulated Quantization Linear Layer (KVQuant-style)
# ============================================================================

class QuantLinearSim(nn.Module):
    """
    Simulated quantization wrapper for Linear layers (for K/V projections).
    This replaces the original k_proj/v_proj and applies fake quantization
    to the output activations.
    """
    def __init__(
        self,
        name,
        bits,
        original_layer,
        perchannel=True,
        include_sparse=False,
        sparsity_threshold=0.999,
        dynamicquantization=True,
        nuq=False,
        first_few_fp16=-1,
        clamp=False
    ):
        super().__init__()
        self.name = name
        self.bits = bits
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Copy weight and bias from original layer
        self.weight = nn.Parameter(original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        
        # Quantization settings
        self.perchannel = perchannel
        self.qchannel = 0 if perchannel else -1  # 0 for per-channel (K), -1 for per-token (V)
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.dynamicquantization = dynamicquantization
        self.nuq = nuq
        self.first_few_fp16 = first_few_fp16
        self.clamp = clamp

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        
        # Compute linear output
        x = x.half()
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        y = y.float()
        
        # Skip quantization if bits >= 16
        if self.bits >= 16:
            return y.reshape(out_shape).half()
        
        # Detect outliers if using dense-and-sparse quantization
        if self.include_sparse:
            outlier_mask = get_outliers_dynamic(
                y,
                channel=self.qchannel,
                thresh=self.sparsity_threshold,
                first_few_fp16=self.first_few_fp16
            )
        else:
            outlier_mask = None
        
        # Apply simulated quantization
        y = quant_fn_zp(
            y,
            bits=self.bits,
            qchannel=self.qchannel,
            include_sparse=self.include_sparse,
            outlier_mask=outlier_mask,
            dynamicquantization=self.dynamicquantization,
            clamp=self.clamp
        )
        
        y = y.reshape(out_shape)
        return y.half()


# ============================================================================
# Model Modification Functions
# ============================================================================

def replace_with_quant_sim(
    model,
    bits,
    perchannel_match=["k_proj"],
    pertoken_match=["v_proj"],
    include_sparse=False,
    sparsity_threshold=0.999,
    first_few_fp16=-1,
    clamp=False
):
    """
    Replace k_proj and v_proj layers with QuantLinearSim for simulated quantization.
    Supports GQA architecture (different number of KV heads vs Q heads).
    """
    replaced_count = 0
    
    for name, module in model.named_modules():
        # Check if this is a layer we want to replace
        is_perchannel = any(p in name for p in perchannel_match)
        is_pertoken = any(p in name for p in pertoken_match)
        
        if (is_perchannel or is_pertoken) and isinstance(module, nn.Linear):
            # Find parent module
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            
            # Create quantized replacement
            quant_layer = QuantLinearSim(
                name=name,
                bits=bits,
                original_layer=module,
                perchannel=is_perchannel,  # per-channel for K, per-token for V
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                dynamicquantization=True,
                first_few_fp16=first_few_fp16,
                clamp=clamp
            )
            
            setattr(parent, attr_name, quant_layer)
            replaced_count += 1
            logger.debug(f"Replaced {name} with QuantLinearSim (bits={bits}, perchannel={is_perchannel})")
    
    logger.info(f"Replaced {replaced_count} layers with simulated quantization")
    return model


# ============================================================================
# Data Loading Functions (from your run_ppl_eval.py)
# ============================================================================

def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    elif "c4" in name:
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
                
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        testenc = TokenizerWrapper(testenc)
        return testenc
    elif "ptb" in name:
        valdata = load_dataset(
            "ptb-text-only/ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    else:
        raise NotImplementedError(f"Dataset {name} not supported")


# ============================================================================
# PPL Evaluation Function (from your run_ppl_eval.py)
# ============================================================================

@torch.no_grad()
def eval_ppl(model, tokenizer, model_name, datasets, seqlen=2048, device="cuda", limit=512):
    model = model.to(device)
    if isinstance(device, str):
        device = torch.device(device)

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
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        nlls = []

        for i in tqdm(range(min(nsamples, limit)), desc=f"Evaluating {dataset}"):
            batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(device)

            outputs = model.model(batch)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][:, 1:].to(device)
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
            
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
            
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
        model.config.use_cache = use_cache
        results[dataset] = ppl.item()

    return results


# ============================================================================
# Model Loading Function
# ============================================================================

def get_model(model_path, seqlen, maxseqlen, use_flash_attn=True):
    """
    Load LLaMA model with proper RoPE scaling for long sequences.
    Supports LLaMA3 with GQA architecture.
    """
    def skip(*args, **kwargs):
        pass
    
    # Skip initialization for faster loading
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # Load config and set RoPE scaling if needed
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and maxseqlen > orig_ctx_len:
        scaling_factor = float(math.ceil(maxseqlen / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        logger.info(f"Applied RoPE scaling factor: {scaling_factor}")

    # Load model
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=torch.float16,
        )
    except Exception as e:
        logger.warning(f"Failed to load with {attn_impl}, falling back to default: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    
    model.seqlen = seqlen
    model.eval()
    
    # Log GQA info
    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", None)
    if num_key_value_heads and num_attention_heads:
        if num_key_value_heads != num_attention_heads:
            logger.info(f"GQA detected: Q heads={num_attention_heads}, KV heads={num_key_value_heads}")
        else:
            logger.info(f"MHA detected: {num_attention_heads} heads")
    
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KVQuant-style simulated quantization with PPL evaluation")
    
    # Model arguments
    parser.add_argument('model', type=str, help='Path to LLaMA model')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length for evaluation')
    parser.add_argument('--maxseqlen', type=int, default=2048, help='Maximum sequence length for RoPE scaling')
    
    # Quantization arguments
    parser.add_argument('--abits', type=int, default=4, choices=[2, 3, 4, 8, 16],
                        help='Number of bits for KV cache quantization (16 = no quant)')
    parser.add_argument('--perchannel', type=str, nargs='+', default=["k_proj"],
                        help='Layers to use per-channel (per-head) quantization')
    parser.add_argument('--pertoken', type=str, nargs='+', default=["v_proj"],
                        help='Layers to use per-token quantization')
    parser.add_argument('--include_sparse', action='store_true',
                        help='Use dense-and-sparse quantization (keep outliers in FP16)')
    parser.add_argument('--sparsity_threshold', type=float, default=0.99,
                        help='Percentile threshold for outlier detection')
    parser.add_argument('--first_few_fp16', type=int, default=-1,
                        help='Keep first N tokens in FP16 (attention sink)')
    parser.add_argument('--clamp', action='store_true',
                        help='Clamp zero-point in integer quantization')
    
    # Evaluation arguments
    parser.add_argument('--datasets', type=str, default='wikitext2',
                        help='Datasets to evaluate (comma-separated)')
    parser.add_argument('--limit', type=int, default=512,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--no_flash_attn', action='store_true',
                        help='Disable flash attention')
    
    # Misc arguments
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--save_quantizer', type=str, default=None,
                        help='Path to save quantizer info (for debugging)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level="DEBUG" if args.verbose else "INFO"
    )
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("KVQuant-style Simulated Quantization for LLaMA3")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Quantization bits: {args.abits}")
    logger.info(f"Per-channel layers: {args.perchannel}")
    logger.info(f"Per-token layers: {args.pertoken}")
    logger.info(f"Include sparse: {args.include_sparse}")
    if args.include_sparse:
        logger.info(f"  Sparsity threshold: {args.sparsity_threshold}")
    if args.first_few_fp16 > 0:
        logger.info(f"Attention sink (first {args.first_few_fp16} tokens in FP16)")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Sequence length: {args.seqlen}")
    logger.info("=" * 60)
    
    # Load model and tokenizer
    logger.info("Loading model...")
    model = get_model(
        args.model,
        args.seqlen,
        args.maxseqlen,
        use_flash_attn=not args.no_flash_attn
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Apply simulated quantization
    if args.abits < 16:
        logger.info(f"Applying {args.abits}-bit simulated quantization...")
        model = replace_with_quant_sim(
            model,
            bits=args.abits,
            perchannel_match=args.perchannel,
            pertoken_match=args.pertoken,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            first_few_fp16=args.first_few_fp16,
            clamp=args.clamp
        )
    else:
        logger.info("No quantization applied (abits=16)")
    
    # Convert to half precision
    model = model.half()
    
    # Run PPL evaluation
    logger.info("Starting PPL evaluation...")
    results = eval_ppl(
        model,
        tokenizer,
        args.model,
        args.datasets,
        seqlen=args.seqlen,
        device=args.device,
        limit=args.limit
    )
    
    # Print results
    logger.info("=" * 60)
    logger.info("Results:")
    logger.info("=" * 60)
    for dataset, ppl in results.items():
        logger.info(f"  {dataset}: PPL = {ppl:.4f}")
    logger.info("=" * 60)
