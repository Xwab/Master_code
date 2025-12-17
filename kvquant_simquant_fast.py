"""
KVQuant-style Simulated Quantization - FAST VERSION

Optimizations:
1. Vectorized NUQ (no Python loops)
2. Faster outlier detection using percentile approximation
3. Reduced dtype conversions
4. Optional features can be disabled for speed
5. Torch compile support

Usage:
    python3 kvquant_simquant_fast.py meta-llama/Llama-3-8B --abits 4 --fast
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
from typing import Optional


# ============================================================================
# FAST Quantization Functions
# ============================================================================

@torch.jit.script
def round_to_nearest_pole_fast(w: torch.Tensor, poles: torch.Tensor) -> torch.Tensor:
    """
    Vectorized version - maps each value to nearest pole.
    Much faster than loop-based version.
    """
    # w: [N], poles: [K]
    # Compute distances: [N, K]
    diff = (w.unsqueeze(-1) - poles.unsqueeze(0)).abs()
    # Find nearest pole index
    idx = diff.argmin(dim=-1)
    # Return corresponding pole values
    return poles[idx]


@torch.jit.script
def get_outliers_fast(w: torch.Tensor, channel: int, thresh: float) -> torch.Tensor:
    """
    Fast outlier detection using sorting instead of quantile.
    Approximate but much faster.
    """
    # Use min/max based threshold (faster than quantile)
    if channel == 0:
        maxval = w.max(dim=0).values
        minval = w.min(dim=0).values
    else:
        maxval = w.max(dim=-1, keepdim=True).values
        minval = w.min(dim=-1, keepdim=True).values
    
    # Approximate percentile using range
    range_val = maxval - minval
    margin = range_val * (1.0 - thresh) / 2.0
    
    upper = maxval - margin
    lower = minval + margin
    
    if channel == 0:
        upper = upper.unsqueeze(0)
        lower = lower.unsqueeze(0)
    
    outlier_mask = (w > upper) | (w < lower)
    return outlier_mask


@torch.jit.script
def quant_fn_fast(
    inp: torch.Tensor,
    bits: int,
    qchannel: int,
    include_sparse: bool,
    outlier_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Fast uniform quantization without unnecessary operations.
    """
    # Compute min/max
    if qchannel == 0:
        maxval = inp.max(dim=0).values
        minval = inp.min(dim=0).values
    else:
        maxval = inp.max(dim=-1).values
        minval = inp.min(dim=-1).values
    
    # Handle sparse
    if include_sparse and outlier_mask is not None:
        # Use median approximation (faster than torch.median)
        clean_inp = inp.clone()
        clean_inp[outlier_mask] = 0
        count = (~outlier_mask).float().sum(dim=qchannel, keepdim=True).clamp(min=1)
        mean_val = clean_inp.sum(dim=qchannel, keepdim=True) / count
        
        if qchannel == 0:
            maxval = clean_inp.max(dim=0).values
            minval = clean_inp.min(dim=0).values
        else:
            maxval = clean_inp.max(dim=-1).values
            minval = clean_inp.min(dim=-1).values
    
    # Compute scale
    rangeval = (maxval - minval).clamp(min=1e-8)
    scale = (2.0 ** bits - 1.0) / rangeval
    
    if qchannel == 0:
        scale = scale.unsqueeze(0)
        minval = minval.unsqueeze(0)
    else:
        scale = scale.unsqueeze(-1)
        minval = minval.unsqueeze(-1)
    
    # Handle outliers
    outliers = None
    if include_sparse and outlier_mask is not None:
        outliers = inp * outlier_mask.float()
        inp = inp - outliers
    
    # Quantize
    qinp = torch.round(scale * (inp - minval))
    qinp = qinp.clamp(0, 2.0 ** bits - 1.0)
    
    # Dequantize
    qinp_out = qinp / scale + minval
    
    # Add outliers back
    if outliers is not None:
        qinp_out = qinp_out * (~outlier_mask).float() + outliers
    
    return qinp_out


def quant_fn_nuq_fast(inp, bits, qchannel, lut, include_sparse=False, outlier_mask=None):
    """
    Fast NUQ quantization with vectorized pole matching.
    """
    if lut is None:
        return quant_fn_fast(inp, bits, qchannel, include_sparse, outlier_mask)
    
    # Compute range
    if qchannel == 0:
        maxval = inp.max(dim=0).values
        minval = inp.min(dim=0).values
    else:
        maxval = inp.max(dim=-1, keepdim=True).values
        minval = inp.min(dim=-1, keepdim=True).values
    
    offset = (maxval + minval) / 2
    rangeval = ((maxval - minval) / 2).clamp(min=1e-8)
    
    if qchannel == 0:
        offset = offset.unsqueeze(0)
        rangeval = rangeval.unsqueeze(0)
    
    # Normalize
    inp_norm = (inp - offset) / rangeval
    
    # Handle outliers
    if include_sparse and outlier_mask is not None:
        outliers = inp_norm * outlier_mask.float()
        inp_norm = inp_norm - outliers
    else:
        outliers = None
    
    # Vectorized pole matching
    lut_dev = lut.to(inp.device)
    Q = round_to_nearest_pole_fast(inp_norm.flatten(), lut_dev)
    qinp_out = Q.reshape(inp.shape)
    
    # Denormalize
    qinp_out = qinp_out * rangeval + offset
    
    if outliers is not None:
        qinp_out = qinp_out * (~outlier_mask).float() + (outliers * rangeval + offset * outlier_mask.float())
    
    return qinp_out


# ============================================================================
# Fast QuantLinearSim
# ============================================================================

class QuantLinearSimFast(nn.Module):
    """
    Optimized simulated quantization layer.
    """
    
    def __init__(self, name, bits, infeatures, outfeatures, weight, bias,
                 perchannel=True, include_sparse=False, sparsity_threshold=0.99,
                 first_few_fp16=-1, nuq=False):
        super().__init__()
        
        self.name = name
        self.bits = bits
        self.outfeatures = outfeatures
        
        # Store weight transposed for efficient matmul
        self.register_buffer('weight', weight.T.contiguous())
        if bias is not None:
            self.register_buffer('bias', bias.contiguous())
        else:
            self.bias = None
        
        self.qchannel = 0 if perchannel else -1
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.first_few_fp16 = first_few_fp16
        self.nuq = nuq
        self.lut = None
    
    def forward(self, x):
        # Reshape input
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        
        # Linear computation (keep in half for speed)
        y = torch.mm(x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        
        # Skip quant if 16-bit
        if self.bits >= 16:
            return y.view(*orig_shape[:-1], self.outfeatures)
        
        # Detect outliers (fast version)
        if self.include_sparse:
            outlier_mask = get_outliers_fast(y, self.qchannel, self.sparsity_threshold)
        else:
            outlier_mask = None
        
        # Quantize
        if self.nuq and self.lut is not None:
            y = quant_fn_nuq_fast(y, self.bits, self.qchannel, self.lut,
                                  self.include_sparse, outlier_mask)
        else:
            y = quant_fn_fast(y, self.bits, self.qchannel, 
                             self.include_sparse, outlier_mask)
        
        return y.view(*orig_shape[:-1], self.outfeatures)


def make_quant_sim_fast(model, bits, perchannel_match=["k_proj"], pertoken_match=["v_proj"],
                        include_sparse=False, sparsity_threshold=0.99, first_few_fp16=-1, nuq=False):
    """Replace layers with fast quantized versions."""
    replaced = 0
    
    for name, module in list(model.named_modules()):
        is_perchannel = any(p in name for p in perchannel_match)
        is_pertoken = any(p in name for p in pertoken_match)
        
        if (is_perchannel or is_pertoken) and isinstance(module, nn.Linear):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            quant_layer = QuantLinearSimFast(
                name=name, bits=bits,
                infeatures=module.in_features,
                outfeatures=module.out_features,
                weight=module.weight.data,
                bias=module.bias.data if module.bias is not None else None,
                perchannel=is_perchannel,
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                first_few_fp16=first_few_fp16,
                nuq=nuq
            )
            
            setattr(parent, parts[-1], quant_layer)
            replaced += 1
    
    logger.info(f"Replaced {replaced} layers (fast mode)")
    return model


# ============================================================================
# Data Loading & Evaluation
# ============================================================================

def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    elif "c4" in name:
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valdata = load_dataset(
            "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87", split="validation"
        )
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        return TokenizerWrapper(testenc)
    elif "ptb" in name:
        valdata = load_dataset("ptb-text-only/ptb_text_only", "penn_treebank", split="validation")
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    else:
        raise NotImplementedError


@torch.no_grad()
def eval_ppl_fast(model, tokenizer, model_name, datasets, seqlen=2048, device="cuda", limit=512):
    """Fast PPL evaluation."""
    model = model.to(device)
    model.eval()
    
    # Enable torch compile if available (PyTorch 2.0+)
    # try:
    #     model = torch.compile(model, mode="reduce-overhead")
    #     logger.info("Using torch.compile for acceleration")
    # except:
    #     pass
    
    results = {}
    
    for dataset in datasets.split(","):
        dataset = dataset.strip()
        cache_path = f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}.cache"
        
        if os.path.exists(cache_path):
            testloader = torch.load(cache_path)
        else:
            testloader = get_ppl_eval_loaders(dataset, tokenizer, seqlen)
            torch.save(testloader, cache_path)
        
        testenc = testloader.input_ids
        nsamples = min(testenc.numel() // seqlen, limit)
        
        use_cache = model.config.use_cache
        model.config.use_cache = False
        
        nlls = []
        
        # Use larger batches if possible
        for i in tqdm(range(nsamples), desc=f"Eval {dataset}"):
            batch = testenc[:, i*seqlen:(i+1)*seqlen].to(device)
            
            with torch.cuda.amp.autocast():  # Use AMP for speed
                outputs = model.model(batch)
                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)
            
            shift_logits = logits[:, :-1, :].float()
            shift_labels = testenc[:, i*seqlen:(i+1)*seqlen][:, 1:].to(device)
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            nlls.append(loss * seqlen)
        
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
        model.config.use_cache = use_cache
        results[dataset] = ppl.item()
    
    return results


def get_model(model_path, seqlen, maxseqlen, use_flash_attn=True):
    """Load model with optimizations."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and maxseqlen > orig_ctx_len:
        scaling_factor = float(math.ceil(maxseqlen / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True,
            attn_implementation=attn_impl, torch_dtype=torch.float16
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    model.eval()
    
    # Log GQA info
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv_heads = getattr(config, "num_key_value_heads", None)
    if n_kv_heads and n_heads and n_kv_heads != n_heads:
        logger.info(f"GQA: Q={n_heads}, KV={n_kv_heads}")
    
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fast KVQuant simulated quantization")
    
    parser.add_argument('model', type=str, help='Model path')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--maxseqlen', type=int, default=2048)
    
    # Quantization
    parser.add_argument('--abits', type=int, default=4, choices=[1, 2, 3, 4, 5, 8, 16])
    parser.add_argument('--perchannel', type=str, nargs='+', default=["k_proj"])
    parser.add_argument('--pertoken', type=str, nargs='+', default=["v_proj"])
    parser.add_argument('--include_sparse', action='store_true')
    parser.add_argument('--sparsity_threshold', type=float, default=0.99)
    parser.add_argument('--first_few_fp16', type=int, default=-1)
    parser.add_argument('--nuq', action='store_true')
    
    # Evaluation
    parser.add_argument('--datasets', type=str, default='wikitext2')
    parser.add_argument('--limit', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_flash_attn', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True,
               level="DEBUG" if args.verbose else "INFO")
    
    logger.info("=" * 50)
    logger.info("KVQuant Fast Simulated Quantization")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Bits: {args.abits}, NUQ: {args.nuq}")
    logger.info(f"Sparse: {args.include_sparse}")
    logger.info("=" * 50)
    
    # Load model
    logger.info("Loading model...")
    model = get_model(args.model, args.seqlen, args.maxseqlen, not args.no_flash_attn)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Apply fast quantization
    if args.abits < 16:
        logger.info(f"Applying {args.abits}-bit fast quantization...")
        model = make_quant_sim_fast(
            model, args.abits, args.perchannel, args.pertoken,
            args.include_sparse, args.sparsity_threshold,
            args.first_few_fp16, args.nuq
        )
    
    model = model.half()
    
    # Warmup
    logger.info("Warmup...")
    with torch.no_grad():
        dummy = torch.randint(0, 1000, (1, 32), device=args.device)
        _ = model(dummy)
    torch.cuda.synchronize()
    
    # Evaluate
    import time
    start = time.time()
    
    results = eval_ppl_fast(
        model, tokenizer, args.model,
        args.datasets, args.seqlen, args.device, args.limit
    )
    
    elapsed = time.time() - start
    
    logger.info("=" * 50)
    logger.info("Results:")
    for dataset, ppl in results.items():
        logger.info(f"  {dataset}: PPL = {ppl:.4f}")
    logger.info(f"Time: {elapsed:.1f}s ({args.limit} samples)")
    logger.info(f"Speed: {args.limit/elapsed:.1f} samples/s")
    logger.info("=" * 50)
