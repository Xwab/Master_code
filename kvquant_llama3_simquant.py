"""
KVQuant-style Simulated Quantization for LLaMA3 (GQA Support)

This script provides both calibration and evaluation modes:
1. Calibration mode: Collect statistics from calibration data and compute quantization parameters
2. Evaluation mode: Load quantization parameters and evaluate PPL

Features:
- Per-channel quantization for Keys (along head dimension)
- Per-token quantization for Values
- Dense-and-sparse quantization (outlier preservation)
- NUQ (Non-Uniform Quantization) support
- Attention sink aware quantization
- Full support for GQA architecture in LLaMA3

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
from typing import Optional, Tuple, Dict, List
import pickle
import numpy as np


# ============================================================================
# Quantization Functions
# ============================================================================

def round_to_nearest_pole_sim(w, poles):
    """Round values to nearest pole (for NUQ)."""
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


def compute_nuq_lut(activations, bits, num_samples=10000):
    """
    Compute NUQ lookup table using K-means clustering.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.warning("sklearn not installed, falling back to uniform quantization")
        return None
    
    # Flatten and sample
    flat = activations.flatten().cpu().numpy()
    if len(flat) > num_samples:
        indices = np.random.choice(len(flat), num_samples, replace=False)
        flat = flat[indices]
    
    # K-means clustering
    n_clusters = 2 ** bits
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(flat.reshape(-1, 1))
    
    # Sort cluster centers
    lut = np.sort(kmeans.cluster_centers_.flatten())
    return torch.tensor(lut, dtype=torch.float32)


def get_outliers_static(w, upper_thresh, lower_thresh, channel=-1, cap_outliers=-1, first_few_fp16=-1):
    """Detect outliers using pre-computed thresholds."""
    upper = upper_thresh.unsqueeze(channel)
    lower = lower_thresh.unsqueeze(channel)
    
    under_lower = w < lower
    above_upper = w > upper
    
    outlier_mask = torch.logical_or(under_lower, above_upper)
    
    if first_few_fp16 > -1:
        outlier_mask[:first_few_fp16, :] = True
    
    return outlier_mask


def get_outliers_dynamic(w, channel=-1, thresh=0.999, first_few_fp16=-1):
    """Dynamically detect outliers."""
    t = 1 - ((1 - thresh) / 2)
    w = w.float()
    
    upper = torch.quantile(w, t, dim=channel)
    lower = torch.quantile(w, 1 - t, dim=channel)
    
    upper = upper.unsqueeze(channel)
    lower = lower.unsqueeze(channel)
    
    under_lower = w <= lower
    above_upper = w >= upper
    
    outlier_mask = torch.logical_or(under_lower, above_upper)
    
    if first_few_fp16 > -1:
        outlier_mask[:first_few_fp16, :] = True
    
    return outlier_mask


def quant_fn_zp(inp, bits=8, qchannel=-1, dynamicquantization=False,
                include_sparse=False, outlier_mask=None, maxval=None, minval=None, clamp=False):
    """Integer quantization with zero-point."""
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

    rangeval = (maxval - minval).clamp(min=1e-8)
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


def quant_fn_nuq(inp, bits=8, qchannel=-1, dynamicquantization=False,
                 include_sparse=False, outlier_mask=None, maxval=None, minval=None, lut=None):
    """Non-uniform quantization using lookup table."""
    if lut is None:
        return quant_fn_zp(inp, bits, qchannel, dynamicquantization, include_sparse, outlier_mask, maxval, minval)
    
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

    # Normalize to [0, 1]
    offset = (maxval + minval) / 2
    rangeval = ((maxval - minval) / 2).clamp(min=1e-8)
    offset = offset.unsqueeze(qchannel)
    rangeval = rangeval.unsqueeze(qchannel)

    inp_norm = (inp - offset) / rangeval

    if include_sparse and outlier_mask is not None:
        outliers = inp_norm * outlier_mask
        inp_norm = inp_norm - outliers

    # Quantize using LUT
    lut = lut.to(inp.device)
    Q = round_to_nearest_pole_sim(inp_norm.flatten(), lut)
    qinp_out = Q.reshape(inp.shape).to(inp.dtype)
    qinp_out = qinp_out * rangeval + offset

    if include_sparse and outlier_mask is not None:
        qinp_out = qinp_out + (outliers * rangeval + offset * outlier_mask.float())

    return torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# Calibration Module (SimQuant)
# ============================================================================

class SimQuant:
    """Collect activation statistics for quantization calibration."""
    
    def __init__(self, layer, bits, perchannel=True, qchannel=0):
        self.layer = layer
        self.bits = bits
        self.perchannel = perchannel
        self.qchannel = qchannel  # 0 for per-channel (K), -1 for per-token (V)
        self.nsamples = 0
        self.activations = []
        
    def add_batch(self, inp, out):
        """Collect output activations."""
        if len(self.activations) < 16:  # Limit memory usage
            self.activations.append(out.detach().cpu().float())
        self.nsamples += out.shape[0]
    
    def quantize(self, include_sparse=False, sparsity_threshold=0.999, nuq=False, 
                 fisher=None, norm=False, cap_outliers=-1, first_few_fp16=-1):
        """Compute quantization parameters from collected activations."""
        if not self.activations:
            return (None, None, None)
        
        # Concatenate all activations
        all_acts = torch.cat(self.activations, dim=0)
        all_acts = all_acts.reshape(-1, all_acts.shape[-1])
        
        # Compute thresholds
        if include_sparse:
            t = 1 - ((1 - sparsity_threshold) / 2)
            upper = torch.quantile(all_acts, t, dim=self.qchannel)
            lower = torch.quantile(all_acts, 1 - t, dim=self.qchannel)
        else:
            upper = torch.max(all_acts, dim=self.qchannel).values
            lower = torch.min(all_acts, dim=self.qchannel).values
        
        # Compute NUQ lookup table if needed
        lut = None
        if nuq:
            lut = compute_nuq_lut(all_acts, self.bits)
        
        return (upper.numpy(), lower.numpy(), lut)
    
    def free(self):
        self.activations = []


# ============================================================================
# Quantized Linear Layer
# ============================================================================

class QuantLinearSim(nn.Module):
    """Simulated quantization for K/V projection outputs."""
    
    def __init__(self, name, bits, original_layer, quantizer_params=None,
                 perchannel=True, include_sparse=False, sparsity_threshold=0.999,
                 dynamicquantization=True, nuq=False, first_few_fp16=-1, clamp=False):
        super().__init__()
        self.name = name
        self.bits = bits
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        self.weight = nn.Parameter(original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        
        self.perchannel = perchannel
        self.qchannel = 0 if perchannel else -1
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.dynamicquantization = dynamicquantization
        self.nuq = nuq
        self.first_few_fp16 = first_few_fp16
        self.clamp = clamp
        
        # Load calibrated parameters if provided
        if quantizer_params is not None:
            self.upper_thresh = torch.tensor(quantizer_params[0]).cuda().half() if quantizer_params[0] is not None else None
            self.lower_thresh = torch.tensor(quantizer_params[1]).cuda().half() if quantizer_params[1] is not None else None
            self.lut = quantizer_params[2]
        else:
            self.upper_thresh = None
            self.lower_thresh = None
            self.lut = None

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        
        x = x.half()
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        y = y.float()
        
        if self.bits >= 16:
            return y.reshape(out_shape).half()
        
        # Detect outliers
        if self.include_sparse:
            if self.dynamicquantization or self.upper_thresh is None:
                outlier_mask = get_outliers_dynamic(
                    y, channel=self.qchannel, thresh=self.sparsity_threshold,
                    first_few_fp16=self.first_few_fp16
                )
            else:
                outlier_mask = get_outliers_static(
                    y, self.upper_thresh.to(y.device), self.lower_thresh.to(y.device),
                    channel=self.qchannel, first_few_fp16=self.first_few_fp16
                )
        else:
            outlier_mask = None
        
        # Apply quantization
        if self.nuq and self.lut is not None:
            y = quant_fn_nuq(
                y, bits=self.bits, qchannel=self.qchannel,
                include_sparse=self.include_sparse, outlier_mask=outlier_mask,
                dynamicquantization=self.dynamicquantization, lut=self.lut
            )
        else:
            y = quant_fn_zp(
                y, bits=self.bits, qchannel=self.qchannel,
                include_sparse=self.include_sparse, outlier_mask=outlier_mask,
                dynamicquantization=self.dynamicquantization, clamp=self.clamp
            )
        
        return y.reshape(out_shape).half()


# ============================================================================
# Model Utilities
# ============================================================================

def find_layers(module, layers=[nn.Linear], name=''):
    """Find all linear layers in the module."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name else name1))
    return res


def replace_with_quant_sim(model, bits, quantizers=None,
                           perchannel_match=["k_proj"], pertoken_match=["v_proj"],
                           include_sparse=False, sparsity_threshold=0.999,
                           dynamicquantization=True, nuq=False, first_few_fp16=-1, clamp=False):
    """Replace K/V projection layers with quantized versions."""
    replaced = 0
    
    for name, module in list(model.named_modules()):
        is_perchannel = any(p in name for p in perchannel_match)
        is_pertoken = any(p in name for p in pertoken_match)
        
        if (is_perchannel or is_pertoken) and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            
            # Get quantizer params if available
            qparams = quantizers.get(name) if quantizers else None
            
            quant_layer = QuantLinearSim(
                name=name, bits=bits, original_layer=module,
                quantizer_params=qparams, perchannel=is_perchannel,
                include_sparse=include_sparse, sparsity_threshold=sparsity_threshold,
                dynamicquantization=dynamicquantization, nuq=nuq,
                first_few_fp16=first_few_fp16, clamp=clamp
            )
            
            setattr(parent, attr_name, quant_layer)
            replaced += 1
            logger.debug(f"Replaced {name} (perchannel={is_perchannel})")
    
    logger.info(f"Replaced {replaced} layers with simulated quantization")
    return model


# ============================================================================
# Data Loading
# ============================================================================

def get_loaders(dataset_name, nsamples, seed, model_path, seqlen):
    """Get calibration and test dataloaders."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if "wikitext2" in dataset_name:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    elif "c4" in dataset_name:
        traindata = load_dataset(
            "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87", split="train"
        )
        valdata = load_dataset(
            "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87", split="validation"
        )
        trainenc = tokenizer(' '.join(traindata[:1100]['text']), return_tensors='pt')
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    else:
        raise NotImplementedError
    
    import random
    random.seed(seed)
    
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        inp = trainenc.input_ids[:, i:i+seqlen]
        trainloader.append(inp)
    
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    
    return trainloader, TokenizerWrapper(testenc.input_ids)


def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    """Get test data for PPL evaluation."""
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
# Calibration
# ============================================================================

@torch.no_grad()
def run_calibration(model, dataloader, dev, perchannel_match, pertoken_match, bits,
                    include_sparse=False, sparsity_threshold=0.999, nuq=False):
    """Run calibration to compute quantization parameters."""
    logger.info("Starting calibration...")
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, 'norm'):
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    nsamples = len(dataloader)
    seqlen = dataloader[0].shape[1]
    hidden_size = model.config.hidden_size
    
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, 'norm'):
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    quantizers = {}
    
    for i in tqdm(range(len(layers)), desc="Calibrating layers"):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        
        # Identify layers to quantize
        perchannel_list = []
        pertoken_list = []
        
        for f in full:
            for p in perchannel_match:
                if p in f:
                    perchannel_list.append(f)
            for p in pertoken_match:
                if p in f:
                    pertoken_list.append(f)
        
        quant_list = perchannel_list + pertoken_list
        
        # Create SimQuant instances
        simquant = {}
        for name in quant_list:
            is_perchannel = name in perchannel_list
            simquant[name] = SimQuant(
                full[name], bits,
                perchannel=is_perchannel,
                qchannel=0 if is_perchannel else -1
            )
        
        # Register hooks
        def add_batch(name):
            def tmp(_, inp, out):
                simquant[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in quant_list:
            handles.append(full[name].register_forward_hook(add_batch(name)))
        
        # Run forward pass
        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]
        
        for h in handles:
            h.remove()
        
        # Compute quantization parameters
        for name in quant_list:
            full_name = f'model.layers.{i}.{name}'
            quantizers[full_name] = simquant[name].quantize(
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                nuq=nuq
            )
            simquant[name].free()
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    model.config.use_cache = use_cache
    logger.info(f"Calibration complete. Computed {len(quantizers)} quantizers.")
    return quantizers


# ============================================================================
# PPL Evaluation
# ============================================================================

@torch.no_grad()
def eval_ppl(model, tokenizer, model_name, datasets, seqlen=2048, device="cuda", limit=512):
    """Evaluate perplexity."""
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
    """Load LLaMA model with proper configuration."""
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
        logger.info(f"Applied RoPE scaling: {scaling_factor}")

    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True,
            attn_implementation=attn_impl, torch_dtype=torch.float16
        )
    except Exception as e:
        logger.warning(f"Flash attention failed, using default: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, trust_remote_code=True, torch_dtype=torch.float16
        )
    
    model.seqlen = seqlen
    model.eval()
    
    # Log architecture info
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv_heads = getattr(config, "num_key_value_heads", None)
    if n_kv_heads and n_heads and n_kv_heads != n_heads:
        logger.info(f"GQA: Q heads={n_heads}, KV heads={n_kv_heads}")
    
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, help='Model path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=16, help='Calibration samples')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--maxseqlen', type=int, default=2048)
    
    # Quantization
    parser.add_argument('--abits', type=int, default=4, choices=[2, 3, 4, 8, 16])
    parser.add_argument('--perchannel', type=str, nargs='+', default=["k_proj"])
    parser.add_argument('--pertoken', type=str, nargs='+', default=["v_proj"])
    parser.add_argument('--include_sparse', action='store_true')
    parser.add_argument('--sparsity_threshold', type=float, default=0.99)
    parser.add_argument('--nuq', action='store_true', help='Use non-uniform quantization')
    parser.add_argument('--first_few_fp16', type=int, default=-1)
    parser.add_argument('--clamp', action='store_true')
    
    # Modes
    parser.add_argument('--calibrate', action='store_true', help='Run calibration')
    parser.add_argument('--quantizer_path', type=str, default=None)
    
    # Evaluation
    parser.add_argument('--datasets', type=str, default='wikitext2')
    parser.add_argument('--limit', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_flash_attn', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Logging
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True,
               level="DEBUG" if args.verbose else "INFO")
    
    DEV = torch.device(args.device)
    
    # Print config
    logger.info("=" * 60)
    logger.info("KVQuant Simulated Quantization for LLaMA3")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Bits: {args.abits}")
    logger.info(f"Per-channel: {args.perchannel}")
    logger.info(f"Per-token: {args.pertoken}")
    logger.info(f"Sparse: {args.include_sparse} (thresh={args.sparsity_threshold})")
    logger.info(f"NUQ: {args.nuq}")
    logger.info("=" * 60)
    
    # Load model
    logger.info("Loading model...")
    model = get_model(args.model, args.seqlen, args.maxseqlen, not args.no_flash_attn)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = model.half()
    
    if args.calibrate:
        # Calibration mode
        dataloader, _ = get_loaders(
            'wikitext2', args.nsamples, args.seed, args.model, args.seqlen
        )
        
        quantizers = run_calibration(
            model, dataloader, DEV,
            args.perchannel, args.pertoken, args.abits,
            args.include_sparse, args.sparsity_threshold, args.nuq
        )
        
        if args.quantizer_path:
            with open(args.quantizer_path, 'wb') as f:
                pickle.dump(quantizers, f)
            logger.info(f"Saved quantizers to {args.quantizer_path}")
    
    else:
        # Evaluation mode
        quantizers = None
        if args.quantizer_path and os.path.exists(args.quantizer_path):
            with open(args.quantizer_path, 'rb') as f:
                quantizers = pickle.load(f)
            logger.info(f"Loaded quantizers from {args.quantizer_path}")
        
        if args.abits < 16:
            model = replace_with_quant_sim(
                model, args.abits, quantizers,
                args.perchannel, args.pertoken,
                args.include_sparse, args.sparsity_threshold,
                dynamicquantization=(quantizers is None),
                nuq=args.nuq, first_few_fp16=args.first_few_fp16,
                clamp=args.clamp
            )
        
        # Evaluate
        results = eval_ppl(
            model, tokenizer, args.model,
            args.datasets, args.seqlen, args.device, args.limit
        )
        
        logger.info("=" * 60)
        logger.info("Results:")
        for dataset, ppl in results.items():
            logger.info(f"  {dataset}: PPL = {ppl:.4f}")
        logger.info("=" * 60)
