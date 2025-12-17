"""
KVQuant-style Simulated Quantization for LLaMA3 with PPL Evaluation

This script EXACTLY replicates KVQuant's simulated quantization approach:
1. Replace k_proj/v_proj with QuantLinearSim
2. K uses per-channel quantization (qchannel=0)
3. V uses per-token quantization (qchannel=-1)
4. Support dense-and-sparse quantization (outlier preservation)

Reference: https://github.com/SqueezeAILab/KVQuant/blob/main/quant/kvquant/simquant_module_quantizer.py
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


# ============================================================================
# Quantization Functions - EXACT copy from KVQuant
# ============================================================================

def get_outliers_dynamic(w, channel=-1, thresh=0.999, first_few_fp16=-1):
    """
    Dynamically detect outliers (from KVQuant simquant_module_quantizer.py)
    
    Args:
        w: activation values (2d matrix)
        channel: which dimension to compute thresholds along
        thresh: percentile for outlier threshold (e.g., 0.99 = 1% outliers)
        first_few_fp16: keep first N tokens in FP16
    """
    t = 1 - ((1 - thresh) / 2)
    w = w.float()
    
    # Compute upper and lower thresholds
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
    Integer quantization with zero-point (EXACT copy from KVQuant)
    
    Args:
        inp: activation values (2d matrix after reshape)
        bits: quantization bits
        qchannel: 0 for per-channel (K), -1 for per-token (V)
        dynamicquantization: compute min/max online
        include_sparse: use dense-and-sparse quantization
        outlier_mask: boolean mask for outlier positions
        maxval/minval: pre-computed thresholds (if not dynamic)
        clamp: whether to clamp zero-point
    """
    # Set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse and outlier_mask is not None:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask
            
            # Recenter using median to avoid outliers skewing distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values
    
    # Compute scale
    rangeval = (maxval - minval)
    qx = (2**bits - 1) / rangeval
    
    # Set offset (zero-point)
    if clamp:
        offset = torch.round(minval * qx)
        offset = offset.clamp(-(2**bits - 1), 0)
    else:
        # This improves accuracy with per-channel key quantization
        offset = minval * qx
    
    offset = offset.unsqueeze(qchannel)
    qx = qx.unsqueeze(qchannel)
    
    # Handle outlier removal before quantization
    if include_sparse and outlier_mask is not None:
        outliers = inp * outlier_mask
        inp = inp - outliers
    
    # Quantize: scale and subtract offset
    qinp = torch.round(qx * inp - offset)
    
    # Clipping
    qinp = torch.clip(qinp, min=0, max=2**bits - 1)
    
    # Dequantize: rescale
    qinp_out = (qinp + offset) / qx
    
    # Add outliers back (they stay in FP16)
    if include_sparse and outlier_mask is not None:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers
    
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)
    return qinp_out


# ============================================================================
# QuantLinearSim - EXACT replication of KVQuant's implementation
# ============================================================================

class QuantLinearSim(nn.Module):
    """
    Simulated quantization for K/V projection layers.
    EXACTLY replicates KVQuant's QuantLinearSim class.
    """
    
    def __init__(
        self,
        name,
        bits,
        infeatures,
        outfeatures,
        weight,
        bias,
        perchannel=True,
        include_sparse=False,
        sparsity_threshold=0.999,
        dynamicquantization=True,
        first_few_fp16=-1,
        clamp=False
    ):
        super().__init__()
        
        self.name = name
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        
        # Store weight TRANSPOSED (same as KVQuant)
        # KVQuant: self.weight = weight.T.detach().cpu()
        self.weight = weight.T.detach().clone()
        
        if bias is not None:
            self.bias = bias.detach().clone()
        else:
            self.bias = None
        
        self.perchannel = perchannel
        self.dynamicquantization = dynamicquantization
        self.clamp = clamp
        
        # qchannel: 0 for per-channel (K), -1 for per-token (V)
        if perchannel:
            self.qchannel = 0
        else:
            self.qchannel = -1
        
        self.ochannel = self.qchannel
        
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.first_few_fp16 = first_few_fp16
    
    def forward(self, x):
        """
        Forward pass with simulated quantization on output.
        EXACTLY matches KVQuant's QuantLinearSim.forward()
        """
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        
        # Move weight to device
        self.weight = self.weight.to(x.device)
        if self.bias is not None:
            self.bias = self.bias.to(x.device)
        
        # Compute linear output (KVQuant: y = x @ self.weight)
        x = x.half()
        y = x @ self.weight  # weight is already transposed
        y = y + self.bias if self.bias is not None else y
        y = y.float()
        
        # Skip quantization if bits >= 16
        if self.bits >= 16:
            y = y.reshape(out_shape)
            return y.half()
        
        # Detect outliers if using dense-and-sparse quantization
        if self.include_sparse:
            outlier_mask = get_outliers_dynamic(
                y,
                channel=self.ochannel,
                thresh=self.sparsity_threshold,
                first_few_fp16=self.first_few_fp16
            )
        else:
            outlier_mask = None
        
        # Apply simulated quantization (integer quantization)
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
# Model Modification - Replace k_proj/v_proj with QuantLinearSim
# ============================================================================

def make_quant_sim(
    model,
    bits,
    perchannel_match=["k_proj"],
    pertoken_match=["v_proj"],
    include_sparse=False,
    sparsity_threshold=0.999,
    dynamicquantization=True,
    first_few_fp16=-1,
    clamp=False
):
    """
    Replace k_proj and v_proj layers with QuantLinearSim.
    Replicates KVQuant's make_quant_sim function.
    """
    replaced_count = 0
    
    for name, module in list(model.named_modules()):
        # Check if this layer should be quantized
        is_perchannel = any(p in name for p in perchannel_match)
        is_pertoken = any(p in name for p in pertoken_match)
        
        if (is_perchannel or is_pertoken) and isinstance(module, nn.Linear):
            # Find parent module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            
            # Create QuantLinearSim replacement
            quant_layer = QuantLinearSim(
                name=name,
                bits=bits,
                infeatures=module.in_features,
                outfeatures=module.out_features,
                weight=module.weight,
                bias=module.bias,
                perchannel=is_perchannel,  # True for K (per-channel), False for V (per-token)
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                dynamicquantization=dynamicquantization,
                first_few_fp16=first_few_fp16,
                clamp=clamp
            )
            
            setattr(parent, attr_name, quant_layer)
            replaced_count += 1
            
            quant_type = "per-channel" if is_perchannel else "per-token"
            logger.debug(f"Replaced {name} -> QuantLinearSim({bits}bit, {quant_type})")
    
    logger.info(f"Replaced {replaced_count} layers with simulated quantization")
    return model


# ============================================================================
# Data Loading (from your run_ppl_eval.py)
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
# PPL Evaluation (from your run_ppl_eval.py)
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
# Model Loading
# ============================================================================

def get_model(model_path, seqlen, maxseqlen, use_flash_attn=True):
    """Load LLaMA model with proper RoPE scaling."""
    def skip(*args, **kwargs):
        pass
    
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # RoPE scaling for long sequences
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and maxseqlen > orig_ctx_len:
        scaling_factor = float(math.ceil(maxseqlen / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        logger.info(f"Applied RoPE scaling factor: {scaling_factor}")

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
        logger.warning(f"Failed with {attn_impl}, fallback to default: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    
    model.seqlen = seqlen
    model.eval()
    
    # Log GQA info
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv_heads = getattr(config, "num_key_value_heads", None)
    if n_kv_heads and n_heads and n_kv_heads != n_heads:
        logger.info(f"GQA detected: Q heads={n_heads}, KV heads={n_kv_heads}")
    
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="KVQuant-style simulated quantization with PPL evaluation"
    )
    
    # Model arguments
    parser.add_argument('model', type=str, help='Path to LLaMA model')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--maxseqlen', type=int, default=2048)
    
    # Quantization arguments (matching KVQuant's interface)
    parser.add_argument('--abits', type=int, default=4, choices=[2, 3, 4, 5, 8, 16],
                        help='Quantization bits for KV cache (16=FP16 baseline)')
    parser.add_argument('--perchannel', type=str, nargs='+', default=["k_proj"],
                        help='Layers using per-channel quantization (default: k_proj)')
    parser.add_argument('--pertoken', type=str, nargs='+', default=["v_proj"],
                        help='Layers using per-token quantization (default: v_proj)')
    parser.add_argument('--include_sparse', action='store_true',
                        help='Use dense-and-sparse quantization (outliers in FP16)')
    parser.add_argument('--sparsity_threshold', type=float, default=0.99,
                        help='Percentile for outlier detection (e.g., 0.99 = 1%% outliers)')
    parser.add_argument('--first_few_fp16', type=int, default=-1,
                        help='Keep first N tokens in FP16 (attention sink)')
    parser.add_argument('--clamp', action='store_true',
                        help='Clamp zero-point in integer quantization')
    
    # Evaluation arguments
    parser.add_argument('--datasets', type=str, default='wikitext2',
                        help='Datasets to evaluate (comma-separated)')
    parser.add_argument('--limit', type=int, default=512,
                        help='Max number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_flash_attn', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
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
    logger.info("KVQuant Simulated Quantization for LLaMA")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Quantization: {args.abits}-bit")
    logger.info(f"  Per-channel (K): {args.perchannel}")
    logger.info(f"  Per-token (V): {args.pertoken}")
    if args.include_sparse:
        outlier_pct = (1 - args.sparsity_threshold) * 100
        logger.info(f"  Dense-and-sparse: {outlier_pct:.1f}% outliers in FP16")
    if args.first_few_fp16 > 0:
        logger.info(f"  Attention sink: first {args.first_few_fp16} tokens in FP16")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Sequence length: {args.seqlen}")
    logger.info("=" * 60)
    
    # Load model
    logger.info("Loading model...")
    model = get_model(
        args.model, args.seqlen, args.maxseqlen,
        use_flash_attn=not args.no_flash_attn
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Apply simulated quantization
    if args.abits < 16:
        logger.info(f"Applying {args.abits}-bit KVQuant simulated quantization...")
        model = make_quant_sim(
            model,
            bits=args.abits,
            perchannel_match=args.perchannel,
            pertoken_match=args.pertoken,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            dynamicquantization=True,
            first_few_fp16=args.first_few_fp16,
            clamp=args.clamp
        )
    else:
        logger.info("No quantization (FP16 baseline)")
    
    model = model.half()
    
    # Run evaluation
    logger.info("Starting PPL evaluation...")
    results = eval_ppl(
        model, tokenizer, args.model,
        args.datasets, args.seqlen, args.device, args.limit
    )
    
    # Print results
    logger.info("=" * 60)
    logger.info("Results:")
    for dataset, ppl in results.items():
        logger.info(f"  {dataset}: PPL = {ppl:.4f}")
    logger.info("=" * 60)
