"""
KVQuant SimQuant PPL Evaluation Script

This script replicates KVQuant's llama_simquant.py workflow:
1. Calibration phase: Collect activation statistics and compute quantization parameters
2. Evaluation phase: Apply simulated quantization using calibrated parameters
"""

import argparse
import logging
import math
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Quantization Functions
# ============================================================================

def get_outliers_dynamic(w, channel=-1, thresh=0.999, first_few_fp16=-1):
    """Dynamically detect outliers using quantile."""
    t = 1 - ((1 - thresh) / 2)
    w_float = w.float()
    
    upper = torch.quantile(w_float, t, dim=channel, keepdim=True)
    lower = torch.quantile(w_float, 1 - t, dim=channel, keepdim=True)
    
    outlier_mask = (w_float <= lower) | (w_float >= upper)
    
    if first_few_fp16 > -1 and outlier_mask.shape[0] > first_few_fp16:
        outlier_mask[:first_few_fp16, :] = True
    
    return outlier_mask


def quant_fn_zp(inp, bits=8, qchannel=-1, include_sparse=False, outlier_mask=None):
    """Integer quantization with zero-point (dynamic)."""
    inp_float = inp.float()
    
    if include_sparse and outlier_mask is not None:
        outliers = inp_float * outlier_mask.float()
        median = torch.median(inp_float, dim=qchannel, keepdim=True).values
        tmp_inp = inp_float * (~outlier_mask).float() + median * outlier_mask.float()
        maxval = torch.max(tmp_inp, dim=qchannel, keepdim=True).values
        minval = torch.min(tmp_inp, dim=qchannel, keepdim=True).values
    else:
        maxval = torch.max(inp_float, dim=qchannel, keepdim=True).values
        minval = torch.min(inp_float, dim=qchannel, keepdim=True).values
        outliers = None
    
    rangeval = (maxval - minval).clamp(min=1e-8)
    scale = (2**bits - 1) / rangeval
    zero_point = minval * scale
    
    if include_sparse and outlier_mask is not None:
        inp_to_quant = inp_float - outliers
    else:
        inp_to_quant = inp_float
    
    qinp = torch.round(scale * inp_to_quant - zero_point)
    qinp = torch.clamp(qinp, 0, 2**bits - 1)
    qinp_out = (qinp + zero_point) / scale
    
    if include_sparse and outlier_mask is not None:
        qinp_out = qinp_out * (~outlier_mask).float() + outliers
    
    return torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# SimQuant Calibration Class
# ============================================================================

class SimQuant:
    """Calibration class for collecting activation statistics."""
    
    def __init__(self, layer, bits, perchannel=True, qchannel=0):
        self.layer = layer
        self.perchannel = perchannel
        self.qchannel = qchannel
        self.bits = bits
        self.rows = layer.weight.shape[0]
        self.nsamples = 0
        self.out = None
    
    def add_batch(self, inp, out):
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        if len(out.shape) == 3:
            out = out.reshape((-1, self.rows))
        self.nsamples += out.shape[0]
        
        if self.out is None:
            self.out = out.float().cpu()
        else:
            self.out = torch.cat((self.out, out.float().cpu()), dim=0)
    
    def quantize(self, include_sparse=False, sparsity_threshold=0.999,
                 nuq=False, first_few_fp16=-1, seqlen=2048):
        """Compute quantization parameters from collected activations."""
        if include_sparse:
            t = 1 - ((1 - sparsity_threshold) / 2)
        else:
            t = 1
        
        data = self.out.numpy()
        
        # Compute thresholds
        upper = np.percentile(data, t * 100, axis=self.qchannel)
        lower = np.percentile(data, (1 - t) * 100, axis=self.qchannel)
        
        centroids = None
        if nuq:
            try:
                from sklearn.cluster import KMeans
                
                # Normalize data
                data_t = torch.tensor(data)
                upper_t = torch.tensor(upper).unsqueeze(self.qchannel)
                lower_t = torch.tensor(lower).unsqueeze(self.qchannel)
                
                rangeval = (upper_t - lower_t) / 2
                offset = (upper_t + lower_t) / 2
                data_norm = ((data_t - offset) / rangeval.clamp(min=1e-8)).numpy()
                
                # Remove outliers for clustering
                mask = (np.abs(data_norm) <= 1).all(axis=1)
                data_clean = data_norm[mask].flatten()
                
                if len(data_clean) > 100000:
                    indices = np.random.choice(len(data_clean), 100000, replace=False)
                    data_clean = data_clean[indices]
                
                kmeans = KMeans(n_clusters=2**self.bits, random_state=0, n_init="auto", max_iter=50)
                kmeans.fit(data_clean.reshape(-1, 1))
                centroids = [kmeans.cluster_centers_.flatten()]
            except Exception as e:
                logger.warning(f"K-means failed: {e}, using uniform quantization")
                centroids = None
        
        return (upper, lower, centroids, None, None)
    
    def free(self):
        self.out = None


# ============================================================================
# QuantLinearSim - Quantized Linear Layer
# ============================================================================

class QuantLinearSim(nn.Module):
    """Simulated quantization for K/V projection layers."""
    
    def __init__(self, name, bits, quantizer, infeatures, outfeatures,
                 weight, bias, perchannel=True, include_sparse=False,
                 sparsity_threshold=0.999, dynamicquantization=True,
                 nuq=False, first_few_fp16=-1):
        super().__init__()
        
        self.name = name
        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.dynamicquantization = dynamicquantization
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.first_few_fp16 = first_few_fp16
        self.nuq = nuq
        
        # Quantization channel
        self.qchannel = 0 if perchannel else -1
        
        # Store weight (transposed for x @ W computation)
        # Use nn.Parameter so it moves with model.to(device)
        self.weight = nn.Parameter(weight.T.clone(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        
        # Store calibrated parameters (will be moved manually in forward)
        self.quantizer = quantizer
        if quantizer is not None:
            self._upper = torch.tensor(quantizer[0]).float()
            self._lower = torch.tensor(quantizer[1]).float()
            self._lut = quantizer[2] if nuq and quantizer[2] is not None else None
        else:
            self._upper = None
            self._lower = None
            self._lut = None
    
    def forward(self, x):
        # Get output shape
        out_shape = x.shape[:-1] + (self.outfeatures,)
        
        # Flatten input
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Ensure everything is on the same device as weight
        device = self.weight.device
        dtype = self.weight.dtype
        
        # Move input to weight's device
        x_flat = x_flat.to(device=device, dtype=dtype)
        
        # Linear computation
        y = torch.mm(x_flat, self.weight)
        if self.bias is not None:
            y = y + self.bias
        
        # Convert to float for quantization
        y = y.float()
        
        # Detect outliers
        outlier_mask = None
        if self.include_sparse:
            outlier_mask = get_outliers_dynamic(
                y, channel=self.qchannel, thresh=self.sparsity_threshold,
                first_few_fp16=self.first_few_fp16
            )
        
        # Apply quantization
        if self.nuq and self._lut is not None:
            y = self._quant_nuq(y, outlier_mask, device)
        else:
            y = quant_fn_zp(y, bits=self.bits, qchannel=self.qchannel,
                           include_sparse=self.include_sparse, outlier_mask=outlier_mask)
        
        # Reshape and convert back to original dtype
        y = y.reshape(out_shape).to(dtype=dtype)
        
        return y
    
    def _quant_nuq(self, y, outlier_mask, device):
        """Non-uniform quantization using LUT."""
        y_float = y.float()
        
        # Compute dynamic range
        if self.include_sparse and outlier_mask is not None:
            outliers = y_float * outlier_mask.float()
            median = torch.median(y_float, dim=self.qchannel, keepdim=True).values
            tmp_y = y_float * (~outlier_mask).float() + median * outlier_mask.float()
            maxval = torch.max(tmp_y, dim=self.qchannel, keepdim=True).values
            minval = torch.min(tmp_y, dim=self.qchannel, keepdim=True).values
        else:
            maxval = torch.max(y_float, dim=self.qchannel, keepdim=True).values
            minval = torch.min(y_float, dim=self.qchannel, keepdim=True).values
            outliers = None
        
        # Normalize to [-1, 1]
        offset = (maxval + minval) / 2
        rangeval = ((maxval - minval) / 2).clamp(min=1e-8)
        
        y_shifted = y_float - offset
        if outliers is not None:
            y_shifted = y_shifted - (outliers - offset * outlier_mask.float())
        
        y_norm = y_shifted / rangeval
        
        # Round to nearest LUT entry
        lut = torch.tensor(self._lut[0], device=device, dtype=torch.float32)
        
        # Vectorized nearest pole computation
        y_flat = y_norm.flatten().unsqueeze(1)  # [N, 1]
        distances = torch.abs(y_flat - lut.unsqueeze(0))  # [N, num_poles]
        indices = distances.argmin(dim=1)
        q_flat = lut[indices]
        q_out = q_flat.reshape(y_norm.shape)
        
        # De-normalize
        q_out = q_out * rangeval + offset
        
        # Add outliers back
        if outliers is not None:
            q_out = q_out * (~outlier_mask).float() + outliers
        
        return torch.nan_to_num(q_out, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# Model Modification Functions
# ============================================================================

def find_layers(module, layers=[nn.Linear], name=''):
    """Find all layers of specified types."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, 
                               name=name + '.' + name1 if name else name1))
    return res


def make_quant_sim(model, quantizers, bits, name='', perchannel=True,
                   include_sparse=False, sparsity_threshold=0.999,
                   dynamicquantization=True, nuq=False, first_few_fp16=-1):
    """Replace layers with QuantLinearSim using calibrated quantizers."""
    if isinstance(model, QuantLinearSim):
        return
    
    for attr in dir(model):
        tmp = getattr(model, attr)
        name1 = name + '.' + attr if name else attr
        if name1 in quantizers and isinstance(tmp, nn.Linear):
            quant_layer = QuantLinearSim(
                name=name1,
                bits=bits,
                quantizer=quantizers[name1],
                infeatures=tmp.in_features,
                outfeatures=tmp.out_features,
                weight=tmp.weight.data,
                bias=tmp.bias.data if tmp.bias is not None else None,
                perchannel=perchannel,
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                dynamicquantization=dynamicquantization,
                nuq=nuq,
                first_few_fp16=first_few_fp16
            )
            setattr(model, attr, quant_layer)
    
    for name1, child in model.named_children():
        make_quant_sim(child, quantizers, bits, 
                      name + '.' + name1 if name else name1,
                      perchannel=perchannel,
                      include_sparse=include_sparse,
                      sparsity_threshold=sparsity_threshold,
                      dynamicquantization=dynamicquantization,
                      nuq=nuq,
                      first_few_fp16=first_few_fp16)


# ============================================================================
# Calibration Function
# ============================================================================

@torch.no_grad()
def llama_calibration(model, dataloader, dev, perchannel_match, pertoken_match,
                      bits, include_sparse=False, sparsity_threshold=0.999,
                      nuq=False, first_few_fp16=-1, nsamples=16):
    """Run calibration to compute quantization parameters."""
    logger.info("Starting calibration...")
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    # Move embeddings to device
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), 
                       dtype=dtype, device=dev)
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
            model(batch[0].to(dev))
        except ValueError:
            pass
        if cache['i'] >= nsamples:
            break
    layers[0] = layers[0].module
    
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    logger.info("Calibrating layers...")
    quantizers = {}
    
    for i in tqdm(range(len(layers)), desc="Calibrating"):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        
        # Find target layers
        target_layers = {}
        for f in full:
            for p in perchannel_match + pertoken_match:
                if p in f:
                    target_layers[f] = full[f]
        
        # Create SimQuant objects
        simquant = {}
        for name, module in target_layers.items():
            is_perchannel = any(p in name for p in perchannel_match)
            qchannel = 0 if is_perchannel else -1
            simquant[name] = SimQuant(module, bits, perchannel=is_perchannel, qchannel=qchannel)
        
        # Register hooks
        def make_hook(name):
            def hook(_, inp, out):
                simquant[name].add_batch(inp[0].data, out.data)
            return hook
        
        handles = []
        for name, module in target_layers.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        
        # Forward pass
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), 
                          attention_mask=attention_mask,
                          position_ids=position_ids)[0]
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Compute quantization parameters
        for name in target_layers:
            quantizers[f'model.layers.{i}.{name}'] = simquant[name].quantize(
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                nuq=nuq,
                first_few_fp16=first_few_fp16,
                seqlen=model.seqlen
            )
            simquant[name].free()
        
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    model.config.use_cache = use_cache
    return quantizers


# ============================================================================
# Evaluation Function
# ============================================================================

@torch.no_grad()
def llama_eval(model, testenc, dev):
    """Evaluate model perplexity."""
    logger.info("Evaluating...")
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), 
                       dtype=dtype, device=dev)
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
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    for i in tqdm(range(len(layers)), desc="Evaluating"):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                          attention_mask=attention_mask,
                          position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * model.seqlen)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    return ppl.item()


# ============================================================================
# Data Loading
# ============================================================================

def get_loaders(dataset_name, nsamples, seed, model_id, seqlen):
    """Get calibration and test dataloaders."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if dataset_name == 'wikitext2':
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    elif dataset_name == 'c4':
        traindata = load_dataset('allenai/c4', 'en', split='train', streaming=True)
        traindata = list(traindata.take(nsamples * 2))
        traintext = "\n\n".join([d['text'] for d in traindata])
        trainenc = tokenizer(traintext, return_tensors='pt')
        
        valdata = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
        valdata = list(valdata.take(256))
        testenc = tokenizer("\n\n".join([d['text'] for d in valdata]), return_tensors='pt')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        trainloader.append((trainenc.input_ids[:, i:i+seqlen],))
    
    return trainloader, testenc


def get_model(model_id, seqlen, maxseqlen):
    """Load model."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    
    if orig_ctx_len and maxseqlen > orig_ctx_len:
        scaling_factor = float(math.ceil(maxseqlen / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, trust_remote_code=True,
        torch_dtype=torch.float16, device_map='cpu'
    )
    model.seqlen = seqlen
    model.eval()
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KVQuant SimQuant PPL Evaluation')
    
    parser.add_argument('model', type=str, help='Model path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=16)
    parser.add_argument('--quantize', action='store_true', help='Run calibration')
    parser.add_argument('--abits', type=int, default=16, choices=[2, 3, 4, 5, 16])
    parser.add_argument('--nuq', action='store_true')
    parser.add_argument('--perchannel', type=str, default='["k_proj"]')
    parser.add_argument('--pertoken', type=str, default='["v_proj"]')
    parser.add_argument('--include_sparse', action='store_true')
    parser.add_argument('--sparsity_threshold', type=float, default=0.99)
    parser.add_argument('--quantizer_path', type=str, default=None)
    parser.add_argument('--first_few_fp16', type=int, default=-1)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--maxseqlen', type=int, default=2048)
    parser.add_argument('--dataset', type=str, default='wikitext2', choices=['wikitext2', 'c4'])
    
    args = parser.parse_args()
    
    import json
    args.perchannel = json.loads(args.perchannel)
    args.pertoken = json.loads(args.pertoken)
    
    DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Bits: {args.abits}")
    logger.info(f"Mode: {'Calibration' if args.quantize else 'Evaluation'}")
    logger.info("=" * 60)
    
    # Load model
    model = get_model(args.model, args.seqlen, args.maxseqlen)
    
    # Load data
    dataloader, testloader = get_loaders(
        args.dataset, args.nsamples, args.seed, args.model, model.seqlen
    )
    
    if args.quantize:
        # Run calibration
        quantizers = llama_calibration(
            model, dataloader, DEV,
            args.perchannel, args.pertoken, args.abits,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            nuq=args.nuq,
            first_few_fp16=args.first_few_fp16,
            nsamples=args.nsamples
        )
        
        if args.quantizer_path:
            with open(args.quantizer_path, 'wb') as f:
                pickle.dump(quantizers, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Quantizers saved to {args.quantizer_path}")
    else:
        if args.quantizer_path is None:
            logger.error("Please provide --quantizer_path")
            exit(1)
        
        with open(args.quantizer_path, 'rb') as f:
            quantizers = pickle.load(f)
        
        # Separate K and V quantizers
        k_quant = {k: v for k, v in quantizers.items() if any(p in k for p in args.perchannel)}
        v_quant = {k: v for k, v in quantizers.items() if any(p in k for p in args.pertoken)}
        
        # Apply quantization - K: per-channel, V: per-token (dynamic)
        logger.info("Applying K quantization (per-channel)...")
        make_quant_sim(model, k_quant, args.abits, perchannel=True,
                      include_sparse=args.include_sparse,
                      sparsity_threshold=args.sparsity_threshold,
                      dynamicquantization=False, nuq=args.nuq,
                      first_few_fp16=args.first_few_fp16)
        
        logger.info("Applying V quantization (per-token, dynamic)...")
        make_quant_sim(model, v_quant, args.abits, perchannel=False,
                      include_sparse=args.include_sparse,
                      sparsity_threshold=args.sparsity_threshold,
                      dynamicquantization=True, nuq=args.nuq,
                      first_few_fp16=args.first_few_fp16)
        
        # Evaluate
        ppl = llama_eval(model, testloader, DEV)
        logger.info("=" * 60)
        logger.info(f"Perplexity: {ppl:.4f}")
        logger.info("=" * 60)
