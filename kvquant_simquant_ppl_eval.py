"""
KVQuant-style Simulated Quantization for LLaMA3 with PPL Evaluation

This script EXACTLY replicates KVQuant's simulated quantization approach:
1. Pre-RoPE quantization: Quantize K/V BEFORE applying RoPE
2. K uses per-channel quantization (qchannel=0)
3. V uses per-token quantization (qchannel=-1)
4. Support dense-and-sparse quantization (outlier preservation)

Two modes:
- --pre_rope: True pre-RoPE quantization (quantize K before RoPE, RoPE applied after)
- Default: Quantize k_proj/v_proj output (matches KVQuant simquant behavior)

Reference: https://github.com/SqueezeAILab/KVQuant
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention,
    apply_rotary_pos_emb, repeat_kv
)
from loguru import logger
import math
from typing import Optional, Tuple


# ============================================================================
# Quantization Functions - EXACT copy from KVQuant
# ============================================================================

def get_outliers_dynamic(w, channel=-1, thresh=0.999, first_few_fp16=-1):
    """
    Dynamically detect outliers (from KVQuant simquant_module_quantizer.py)
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
    dynamicquantization=True,
    include_sparse=False,
    outlier_mask=None,
    maxval=None,
    minval=None,
    clamp=False
):
    """
    Integer quantization with zero-point (EXACT copy from KVQuant)
    """
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
    
    rangeval = (maxval - minval)
    qx = (2**bits - 1) / rangeval
    
    if clamp:
        offset = torch.round(minval * qx)
        offset = offset.clamp(-(2**bits - 1), 0)
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
    
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)
    return qinp_out


def simulated_quantize(
    tensor,
    bits,
    qchannel,
    include_sparse=False,
    sparsity_threshold=0.999,
    first_few_fp16=-1,
    clamp=False
):
    """
    Apply simulated quantization to a tensor.
    
    Args:
        tensor: Input tensor [batch, seq, dim] or [batch, heads, seq, head_dim]
        bits: Quantization bits
        qchannel: Quantization channel (0=per-channel, -1=per-token)
        include_sparse: Use dense-and-sparse quantization
        sparsity_threshold: Outlier threshold
        first_few_fp16: Keep first N tokens in FP16
    """
    if bits >= 16:
        return tensor
    
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    
    # Reshape to 2D for quantization
    tensor = tensor.reshape(-1, tensor.shape[-1]).float()
    
    # Detect outliers
    if include_sparse:
        outlier_mask = get_outliers_dynamic(
            tensor, channel=qchannel, thresh=sparsity_threshold,
            first_few_fp16=first_few_fp16
        )
    else:
        outlier_mask = None
    
    # Quantize
    tensor = quant_fn_zp(
        tensor, bits=bits, qchannel=qchannel,
        include_sparse=include_sparse, outlier_mask=outlier_mask,
        dynamicquantization=True, clamp=clamp
    )
    
    return tensor.reshape(orig_shape).to(orig_dtype)


# ============================================================================
# QuantLinearSim - For k_proj/v_proj replacement (simquant mode)
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
        first_few_fp16=-1,
        clamp=False
    ):
        super().__init__()
        
        self.name = name
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        
        # Store weight TRANSPOSED (same as KVQuant)
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
        
        # Detect outliers
        if self.include_sparse:
            outlier_mask = get_outliers_dynamic(
                y, channel=self.qchannel, thresh=self.sparsity_threshold,
                first_few_fp16=self.first_few_fp16
            )
        else:
            outlier_mask = None
        
        # Quantize
        y = quant_fn_zp(
            y, bits=self.bits, qchannel=self.qchannel,
            include_sparse=self.include_sparse, outlier_mask=outlier_mask,
            dynamicquantization=True, clamp=self.clamp
        )
        
        return y.reshape(out_shape).half()


# ============================================================================
# Pre-RoPE Quantization Attention (True KVQuant deployment style)
# ============================================================================

class PreRoPEQuantAttention(nn.Module):
    """
    Attention module with TRUE Pre-RoPE quantization.
    
    Quantization happens BEFORE RoPE is applied to keys:
    1. key_states = k_proj(hidden_states)
    2. key_states = QUANTIZE(key_states)  <- Pre-RoPE
    3. key_states = apply_rope(key_states)
    
    This matches KVQuant's deployment kernel behavior.
    """
    
    def __init__(self, original_attn, bits, include_sparse=False, 
                 sparsity_threshold=0.999, first_few_fp16=-1, clamp=False):
        super().__init__()
        
        # Copy all attributes from original attention
        self.config = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.attention_dropout = original_attn.attention_dropout
        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.num_key_value_heads = original_attn.num_key_value_heads
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.max_position_embeddings = original_attn.max_position_embeddings
        self.rope_theta = original_attn.rope_theta
        self.is_causal = original_attn.is_causal
        
        # Copy layers
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.rotary_emb = original_attn.rotary_emb
        
        # Quantization settings
        self.bits = bits
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.first_few_fp16 = first_few_fp16
        self.clamp = clamp
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        
        # Compute Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # ========== PRE-ROPE QUANTIZATION ==========
        # Quantize K BEFORE applying RoPE (this is the key difference!)
        # K: per-channel quantization (qchannel=0 in flattened view means per-head-dim)
        key_states_flat = key_states.transpose(1, 2).reshape(bsz * q_len, -1)  # [B*T, H*D]
        key_states_flat = simulated_quantize(
            key_states_flat, self.bits, qchannel=0,  # per-channel
            include_sparse=self.include_sparse,
            sparsity_threshold=self.sparsity_threshold,
            first_few_fp16=self.first_few_fp16,
            clamp=self.clamp
        )
        key_states = key_states_flat.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # V: per-token quantization
        value_states_flat = value_states.transpose(1, 2).reshape(bsz * q_len, -1)  # [B*T, H*D]
        value_states_flat = simulated_quantize(
            value_states_flat, self.bits, qchannel=-1,  # per-token
            include_sparse=self.include_sparse,
            sparsity_threshold=self.sparsity_threshold,
            first_few_fp16=self.first_few_fp16,
            clamp=self.clamp
        )
        value_states = value_states_flat.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # ============================================
        
        # Get RoPE embeddings
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        
        # Apply RoPE AFTER quantization
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


# ============================================================================
# Model Modification Functions
# ============================================================================

def make_quant_sim(model, bits, perchannel_match=["k_proj"], pertoken_match=["v_proj"],
                   include_sparse=False, sparsity_threshold=0.999, first_few_fp16=-1, clamp=False):
    """Replace k_proj/v_proj with QuantLinearSim (simquant mode)."""
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
    
    logger.info(f"Replaced {replaced} layers with QuantLinearSim (simquant mode)")
    return model


def make_pre_rope_quant(model, bits, include_sparse=False, sparsity_threshold=0.999,
                        first_few_fp16=-1, clamp=False):
    """Replace attention modules with PreRoPEQuantAttention (true pre-RoPE mode)."""
    replaced = 0
    
    for name, module in list(model.named_modules()):
        if isinstance(module, (LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention)):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            
            quant_attn = PreRoPEQuantAttention(
                module, bits, include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                first_few_fp16=first_few_fp16, clamp=clamp
            )
            
            setattr(parent, attr_name, quant_attn)
            replaced += 1
    
    logger.info(f"Replaced {replaced} attention modules with PreRoPEQuantAttention")
    return model


# ============================================================================
# Data Loading
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


# ============================================================================
# PPL Evaluation
# ============================================================================

@torch.no_grad()
def eval_ppl(model, tokenizer, model_name, datasets, seqlen=2048, device="cuda", limit=512):
    model = model.to(device)
    device = torch.device(device) if isinstance(device, str) else device
    results = {}

    for dataset in datasets.split(","):
        dataset = dataset.strip()
        cache_path = f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
        
        if os.path.exists(cache_path):
            testloader = torch.load(cache_path)
        else:
            testloader = get_ppl_eval_loaders(dataset, tokenizer, seqlen)
            torch.save(testloader, cache_path)
        
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
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            nlls.append(loss.float() * seqlen)
        
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
        model.config.use_cache = use_cache
        results[dataset] = ppl.item()

    return results


# ============================================================================
# Model Loading
# ============================================================================

def get_model(model_path, seqlen, maxseqlen, use_flash_attn=True):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and maxseqlen > orig_ctx_len:
        scaling_factor = float(math.ceil(maxseqlen / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # For pre-RoPE mode, we need eager attention (not flash/sdpa)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
            torch_dtype=torch.float16
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True, torch_dtype=torch.float16
        )
    
    model.seqlen = seqlen
    model.eval()
    
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv_heads = getattr(config, "num_key_value_heads", None)
    if n_kv_heads and n_heads and n_kv_heads != n_heads:
        logger.info(f"GQA: Q heads={n_heads}, KV heads={n_kv_heads}")
    
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KVQuant-style simulated quantization")
    
    parser.add_argument('model', type=str, help='Path to LLaMA model')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--maxseqlen', type=int, default=2048)
    
    # Quantization
    parser.add_argument('--abits', type=int, default=4, choices=[2, 3, 4, 5, 8, 16])
    parser.add_argument('--perchannel', type=str, nargs='+', default=["k_proj"])
    parser.add_argument('--pertoken', type=str, nargs='+', default=["v_proj"])
    parser.add_argument('--include_sparse', action='store_true')
    parser.add_argument('--sparsity_threshold', type=float, default=0.99)
    parser.add_argument('--first_few_fp16', type=int, default=-1)
    parser.add_argument('--clamp', action='store_true')
    
    # Mode selection
    parser.add_argument('--pre_rope', action='store_true',
                        help='Use TRUE pre-RoPE quantization (quantize before RoPE)')
    
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
    
    logger.info("=" * 60)
    logger.info("KVQuant Simulated Quantization")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Bits: {args.abits}")
    if args.pre_rope:
        logger.info("Mode: TRUE Pre-RoPE quantization")
    else:
        logger.info("Mode: SimQuant (k_proj/v_proj output quantization)")
    if args.include_sparse:
        logger.info(f"Sparse: {(1-args.sparsity_threshold)*100:.1f}% outliers in FP16")
    logger.info("=" * 60)
    
    # Load model
    logger.info("Loading model...")
    # For pre_rope mode, disable flash attention
    use_flash = not args.no_flash_attn and not args.pre_rope
    model = get_model(args.model, args.seqlen, args.maxseqlen, use_flash_attn=use_flash)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Apply quantization
    if args.abits < 16:
        if args.pre_rope:
            logger.info("Applying TRUE pre-RoPE quantization...")
            model = make_pre_rope_quant(
                model, args.abits, args.include_sparse,
                args.sparsity_threshold, args.first_few_fp16, args.clamp
            )
        else:
            logger.info("Applying simquant quantization...")
            model = make_quant_sim(
                model, args.abits, args.perchannel, args.pertoken,
                args.include_sparse, args.sparsity_threshold,
                args.first_few_fp16, args.clamp
            )
    
    model = model.half()
    
    # Evaluate
    logger.info("Starting evaluation...")
    results = eval_ppl(model, tokenizer, args.model, args.datasets, args.seqlen, args.device, args.limit)
    
    logger.info("=" * 60)
    logger.info("Results:")
    for dataset, ppl in results.items():
        logger.info(f"  {dataset}: PPL = {ppl:.4f}")
    logger.info("=" * 60)
