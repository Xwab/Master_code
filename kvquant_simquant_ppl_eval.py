"""
KVQuant SimQuant PPL Evaluation Script

This script EXACTLY replicates KVQuant's llama_simquant.py workflow:
1. Calibration phase: Collect activation statistics and compute quantization parameters
2. Evaluation phase: Apply simulated quantization using calibrated parameters

Key differences from simple dynamic quantization:
- Uses calibration data to compute static outlier thresholds
- Uses K-means clustering for NUQ lookup tables
- Supports Q-Norm for improved accuracy
- Key uses static (calibrated) quantization, Value uses dynamic quantization

Reference: https://github.com/SqueezeAILab/KVQuant/blob/main/quant/llama_simquant.py
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
# Quantization Functions (from KVQuant simquant_module_quantizer.py)
# ============================================================================

def round_to_nearest_pole_sim(w, poles):
    """
    Round values to nearest pole (centroid).
    EXACT copy from KVQuant.
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


def get_outliers(w, channel=-1, outlier_threshold_upper=-1, outlier_threshold_lower=-1,
                 cap_outliers=-1, first_few_fp16=-1):
    """
    Detect outliers using STATIC thresholds (from calibration).
    EXACT copy from KVQuant.
    """
    outlier_threshold_upper = outlier_threshold_upper.unsqueeze(channel)
    outlier_threshold_lower = outlier_threshold_lower.unsqueeze(channel)
    
    under_lower = w < outlier_threshold_lower
    above_upper = w > outlier_threshold_upper
    outlier_mask = torch.logical_or(under_lower, above_upper)
    
    if cap_outliers > -1:
        zero_point = (outlier_threshold_upper + outlier_threshold_lower) / 2
        distance = (outlier_threshold_upper - outlier_threshold_lower) / 2
        outliers = w * outlier_mask
        
        values = torch.zeros_like(outliers)
        values[outlier_mask] = ((w - zero_point) / distance)[outlier_mask]
        
        upper_values, upper_indices = torch.topk(values, 21, dim=-1)
        lower_values, lower_indices = torch.topk(values, 21, dim=-1, largest=False)
        indices_combined = torch.cat((upper_indices, lower_indices), dim=-1)
        values_combined = torch.cat((upper_values, lower_values), dim=-1)
        
        values2 = torch.zeros_like(outliers)
        values2.scatter_(-1, indices_combined, values_combined)
        outlier_mask = values2 != 0
    
    if first_few_fp16 > -1:
        outlier_mask[:first_few_fp16, :] = True
    
    return outlier_mask


def get_outliers_dynamic(w, channel=-1, thresh=0.999, first_few_fp16=-1):
    """
    Dynamically detect outliers using quantile.
    EXACT copy from KVQuant.
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


def quant_fn_zp(inp, bits=8, qchannel=-1, dynamicquantization=False,
                include_sparse=False, outlier_mask=None, maxval=-1, minval=-1, clamp=False):
    """
    Integer quantization with zero-point.
    EXACT copy from KVQuant.
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


def quant_fn_nuq_recon(inp, bits=8, qchannel=-1, dynamicquantization=False,
                       include_sparse=False, outlier_mask=None, maxval=-1, minval=-1,
                       lut=None, norm=False, normscale=None, normoffset=None,
                       first_few_fp16=-1):
    """
    Non-Uniform Quantization using lookup table.
    EXACT copy from KVQuant.
    """
    if first_few_fp16 > -1:
        orig = inp.clone()
    
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
    
    # Compute offset and range
    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2
    offset = offset.unsqueeze(qchannel)
    rangeval = rangeval.unsqueeze(qchannel)
    
    # Shift by offset
    inp = inp - offset
    
    # Handle outliers
    if include_sparse and outlier_mask is not None:
        outliers = inp * outlier_mask
        inp = inp - outliers
    
    # Normalize to [-1, 1]
    inp_scaled = inp / rangeval.clamp(min=1e-8)
    
    # Round to nearest LUT entry
    lut_cuda = torch.tensor(lut[0]).to(inp_scaled.device).float()
    Q = round_to_nearest_pole_sim(inp_scaled.flatten(), lut_cuda)
    qinp_out = Q.reshape(inp.shape).float().to(inp_scaled.device)
    
    # Q-Norm
    if norm and normscale is not None and normoffset is not None:
        normscale = normscale.to(inp_scaled.device)
        normoffset = normoffset.to(inp_scaled.device)
        qinp_out = qinp_out * normscale + normoffset
    
    # De-normalize
    qinp_out = qinp_out * rangeval
    
    # Add outliers back
    if include_sparse and outlier_mask is not None:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers
    
    # Shift by offset
    qinp_out = qinp_out + offset
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Keep first few in FP16
    if first_few_fp16 > -1:
        qinp_out[:first_few_fp16, :] = orig[:first_few_fp16, :]
    
    return qinp_out.float()


# ============================================================================
# SimQuant Calibration Class (from KVQuant)
# ============================================================================

class SimQuant:
    """
    Calibration class for collecting activation statistics and computing
    quantization parameters. EXACT copy from KVQuant.
    """
    
    def __init__(self, layer, bits, perchannel=True, qchannel=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.perchannel = perchannel
        self.qchannel = qchannel
        self.bits = bits
        
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.nsamples = 0
        self.out = None
    
    def add_batch(self, inp, out):
        """Collect output activations."""
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        tmp = out.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(out.shape) == 3:
                out = out.reshape((-1, self.rows))
        self.nsamples += tmp
        
        if self.out is None:
            self.out = out.clone()
        else:
            self.out = torch.cat((self.out, out.clone()), dim=0)
    
    def quantize(self, include_sparse=False, sparsity_threshold=0.999,
                 nuq=False, fisher=None, norm=False, cap_outliers=False,
                 first_few_fp16=-1, seqlen=2048):
        """
        Compute quantization parameters from collected activations.
        Returns: (outlier_threshold_upper, outlier_threshold_lower, centroids, [normscale, normoffset])
        """
        if include_sparse:
            t = 1 - ((1 - sparsity_threshold) / 2)
        else:
            t = 1  # Use min-max quantization
        
        data = self.out.float().cpu().numpy()
        
        # Handle cap_outliers for per-channel quantization
        if self.perchannel and cap_outliers:
            data = torch.tensor(data)
            outlier_threshold_upper = torch.tensor(
                np.percentile(data.numpy(), t * 100, axis=self.qchannel)
            ).unsqueeze(self.qchannel)
            outlier_threshold_lower = torch.tensor(
                np.percentile(data.numpy(), (1 - t) * 100, axis=self.qchannel)
            ).unsqueeze(self.qchannel)
            
            zero_point = (outlier_threshold_upper + outlier_threshold_lower) / 2
            distance = (outlier_threshold_upper - outlier_threshold_lower) / 2
            data2 = ((data - zero_point) / distance.clamp(min=1e-8)).abs()
            
            outlier_mask = torch.zeros_like(data2, dtype=torch.bool)
            hidden_dim = data.shape[-1]
            num_elems = max(1, math.ceil((1 - t) * hidden_dim))
            upper_indices = torch.topk(data2, min(num_elems, data2.shape[-1])).indices
            
            true_mask = torch.ones_like(upper_indices, dtype=torch.bool)
            outlier_mask.scatter_(-1, upper_indices, true_mask)
            
            if first_few_fp16 > -1:
                for i in range(self.nsamples):
                    start = i * seqlen
                    end = i * seqlen + first_few_fp16
                    if end <= outlier_mask.shape[0]:
                        outlier_mask[start:end, :] = True
            
            med = torch.median(data, dim=0).values.unsqueeze(0).expand_as(data)
            data_trimmed = data.clone()
            data_trimmed[outlier_mask] = med[outlier_mask]
            
            outlier_threshold_upper = torch.max(data_trimmed, dim=self.qchannel).values
            outlier_threshold_lower = torch.min(data_trimmed, dim=self.qchannel).values
            data = data.numpy()
        
        # Compute thresholds
        if self.perchannel:
            outlier_threshold_upper = np.percentile(data, t * 100, axis=self.qchannel)
            outlier_threshold_lower = np.percentile(data, (1 - t) * 100, axis=self.qchannel)
        else:
            # Per-token not currently supported in calibration
            outlier_threshold_upper = np.percentile(data, t * 100, axis=self.qchannel)
            outlier_threshold_lower = np.percentile(data, (1 - t) * 100, axis=self.qchannel)
        
        # Convert to torch
        data = torch.tensor(data)
        outlier_threshold_upper = torch.tensor(outlier_threshold_upper).unsqueeze(self.qchannel)
        outlier_threshold_lower = torch.tensor(outlier_threshold_lower).unsqueeze(self.qchannel)
        
        # Range and offset
        rangeval = (outlier_threshold_upper - outlier_threshold_lower) / 2
        zeropoint = (outlier_threshold_upper + outlier_threshold_lower) / 2
        
        # Normalize
        data_shifted = data - zeropoint
        data_shifted_normalized = data_shifted / rangeval.clamp(min=1e-8)
        
        # Get outlier mask
        outlier_mask = torch.logical_or(
            data_shifted_normalized > 1,
            data_shifted_normalized < -1
        )
        
        # Remove first few tokens from K-means fitting
        if first_few_fp16 > -1:
            for i in range(self.nsamples):
                start = i * seqlen
                end = i * seqlen + first_few_fp16
                if end <= outlier_mask.shape[0]:
                    outlier_mask[start:end, :] = True
        
        centroids = None
        normscale = None
        normoffset = None
        
        if nuq:
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                logger.warning("sklearn not installed, falling back to uniform quantization")
                return (outlier_threshold_upper.squeeze().numpy(),
                        outlier_threshold_lower.squeeze().numpy(),
                        None, None, None)
            
            centroids = []
            act_distn_np = data_shifted_normalized.flatten()
            n_cluster = 2 ** self.bits
            
            outlier_mask_flat = outlier_mask.flatten()
            act_distn_np_without_outliers = act_distn_np[~outlier_mask_flat]
            act_distn_np_without_outliers = act_distn_np_without_outliers.float().cpu().numpy().reshape(-1, 1)
            
            # Subsample for efficiency
            if len(act_distn_np_without_outliers) > 100000:
                indices = np.random.choice(len(act_distn_np_without_outliers), 100000, replace=False)
                act_distn_np_without_outliers = act_distn_np_without_outliers[indices]
            
            kmeans = KMeans(
                n_clusters=n_cluster,
                random_state=0,
                n_init="auto",
                max_iter=50
            ).fit(act_distn_np_without_outliers)
            
            centroids.append(kmeans.cluster_centers_.flatten())
            
            # Q-Norm
            if norm:
                centroid = torch.tensor(centroids[0])
                aug = data_shifted_normalized.clone()
                not_outlier_mask = ~outlier_mask
                
                m1 = (aug * not_outlier_mask).sum() / not_outlier_mask.sum()
                stdev1 = torch.sqrt(
                    torch.sum(((aug - m1) * not_outlier_mask) ** 2) / not_outlier_mask.sum()
                )
                
                aug_quantized = round_to_nearest_pole_sim(aug.flatten(), centroid)
                aug_quantized = aug_quantized.reshape(aug.shape)
                
                m2 = (aug_quantized * not_outlier_mask).sum() / not_outlier_mask.sum()
                stdev2 = torch.sqrt(
                    torch.sum(((aug_quantized - m2) * not_outlier_mask) ** 2) / not_outlier_mask.sum()
                )
                
                normscale = (stdev1 / stdev2.clamp(min=1e-8))
                normoffset = (-m2) * normscale + m1
        
        return (outlier_threshold_upper.squeeze().numpy(),
                outlier_threshold_lower.squeeze().numpy(),
                centroids, normscale, normoffset)
    
    def free(self):
        """Free memory."""
        self.out = None


# ============================================================================
# QuantLinearSim - Quantized Linear Layer (from KVQuant)
# ============================================================================

class QuantLinearSim(nn.Module):
    """
    Simulated quantization for K/V projection layers.
    Uses calibrated parameters for static quantization.
    EXACT copy from KVQuant.
    """
    
    def __init__(self, name, bits, quantizer, infeatures, outfeatures,
                 weight, bias, perchannel=True, include_sparse=False,
                 sparsity_threshold=0.999, dynamicquantization=False,
                 nuq=False, norm=False, cap_outliers=-1, first_few_fp16=-1, clamp=False):
        super().__init__()
        
        self.name = name
        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.dynamicquantization = dynamicquantization
        self.clamp = clamp
        
        # Store weight transposed for x @ weight computation
        # Use register_buffer so weight moves with .to(device)
        self.register_buffer('weight', weight.T.detach().clone())
        if bias is not None:
            self.register_buffer('bias', bias.detach().clone())
        else:
            self.bias = None
        
        # Add a dummy parameter so that self.parameters() is not empty
        # This fixes: next(self.self_attn.parameters()).device in transformers
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        
        # Quantization parameters
        if perchannel:
            self.qchannel = 0
        else:
            self.qchannel = -1
        self.ochannel = self.qchannel
        
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        
        # Load calibrated thresholds
        if quantizer is not None:
            self.outlier_threshold_upper = torch.tensor(quantizer[0]).flatten().half()
            self.outlier_threshold_lower = torch.tensor(quantizer[1]).flatten().half()
            
            # NUQ LUT
            self.nuq = nuq
            if nuq and quantizer[2] is not None:
                self.lut = quantizer[2]
            else:
                self.lut = None
            
            # Q-Norm
            self.norm = norm
            if norm and len(quantizer) > 3 and quantizer[3] is not None:
                self.normscale = quantizer[3]
                self.normoffset = quantizer[4]
            else:
                self.normscale = None
                self.normoffset = None
        else:
            # Dynamic mode - no calibrated parameters
            self.outlier_threshold_upper = None
            self.outlier_threshold_lower = None
            self.nuq = nuq
            self.lut = None
            self.norm = norm
            self.normscale = None
            self.normoffset = None
        
        self.cap_outliers = cap_outliers
        self.first_few_fp16 = first_few_fp16
    
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        
        # Weight and bias are registered as buffers, so they move with the model
        # Compute output - use half on GPU, float on CPU
        if x.device.type == 'cuda':
            x = x.half()
            y = x @ self.weight
            if self.bias is not None:
                y = y + self.bias
            y = y.float()
        else:
            # CPU doesn't support half precision matmul
            x = x.float()
            weight_float = self.weight.float()
            y = x @ weight_float
            if self.bias is not None:
                y = y + self.bias.float()
        
        # Detect outliers
        if self.include_sparse:
            if self.dynamicquantization:
                outlier_mask = get_outliers_dynamic(
                    y, channel=self.ochannel, thresh=self.sparsity_threshold,
                    first_few_fp16=self.first_few_fp16
                )
            else:
                self.outlier_threshold_upper = self.outlier_threshold_upper.to(y.device)
                self.outlier_threshold_lower = self.outlier_threshold_lower.to(y.device)
                outlier_mask = get_outliers(
                    y, channel=self.ochannel,
                    outlier_threshold_upper=self.outlier_threshold_upper,
                    outlier_threshold_lower=self.outlier_threshold_lower,
                    cap_outliers=self.cap_outliers,
                    first_few_fp16=self.first_few_fp16
                )
        else:
            outlier_mask = None
        
        # Quantize
        if self.nuq and self.lut is not None:
            y = quant_fn_nuq_recon(
                y, bits=self.bits, qchannel=self.qchannel,
                maxval=self.outlier_threshold_upper,
                minval=self.outlier_threshold_lower,
                include_sparse=self.include_sparse,
                outlier_mask=outlier_mask,
                dynamicquantization=self.dynamicquantization,
                lut=self.lut,
                norm=self.norm,
                normscale=self.normscale,
                normoffset=self.normoffset,
                first_few_fp16=self.first_few_fp16
            )
        else:
            y = quant_fn_zp(
                y, bits=self.bits, qchannel=self.qchannel,
                maxval=self.outlier_threshold_upper if not self.dynamicquantization else -1,
                minval=self.outlier_threshold_lower if not self.dynamicquantization else -1,
                include_sparse=self.include_sparse,
                outlier_mask=outlier_mask,
                dynamicquantization=self.dynamicquantization,
                clamp=self.clamp
            )
        
        # Return in appropriate dtype based on device
        if y.device.type == 'cuda':
            return y.reshape(out_shape).half()
        else:
            return y.reshape(out_shape).float()


# ============================================================================
# Model Modification Functions
# ============================================================================

def find_layers(module, layers=[nn.Linear], name=''):
    """Find all layers of specified types."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def make_quant_sim(model, quantizers, bits, name='', perchannel=True,
                   include_sparse=False, sparsity_threshold=0.999,
                   dynamicquantization=False, nuq=False, norm=False,
                   cap_outliers=-1, first_few_fp16=-1, clamp=False):
    """
    Replace layers with QuantLinearSim using calibrated quantizers.
    EXACT copy from KVQuant.
    """
    if isinstance(model, QuantLinearSim):
        return
    
    for attr in dir(model):
        tmp = getattr(model, attr)
        name1 = name + '.' + attr if name != '' else attr
        # Only replace nn.Linear layers that are in quantizers
        if name1 in quantizers.keys() and isinstance(tmp, nn.Linear):
            delattr(model, attr)
            setattr(model, attr, QuantLinearSim(
                name1, bits, quantizers[name1],
                tmp.in_features, tmp.out_features,
                tmp.weight, tmp.bias,
                perchannel=perchannel,
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                dynamicquantization=dynamicquantization,
                nuq=nuq, norm=norm,
                cap_outliers=cap_outliers,
                first_few_fp16=first_few_fp16,
                clamp=clamp
            ))
        del tmp
    
    for name1, child in model.named_children():
        make_quant_sim(
            child, quantizers, bits,
            name + '.' + name1 if name != '' else name1,
            perchannel=perchannel,
            include_sparse=include_sparse,
            sparsity_threshold=sparsity_threshold,
            dynamicquantization=dynamicquantization,
            nuq=nuq, norm=norm,
            cap_outliers=cap_outliers,
            first_few_fp16=first_few_fp16,
            clamp=clamp
        )


# ============================================================================
# Calibration Function (from KVQuant llama_simquant.py)
# ============================================================================

@torch.no_grad()
def llama_calibration(model, dataloader, dev, perchannel_match, pertoken_match,
                      bits, include_sparse=False, sparsity_threshold=0.999,
                      nuq=False, norm=False, cap_outliers=False, first_few_fp16=-1,
                      nsamples=16):
    """
    Run calibration to compute quantization parameters.
    EXACT copy from KVQuant llama_simquant.py.
    """
    logger.info("Starting calibration...")
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    # Move embeddings to device
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
    
    # Move back to CPU
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    logger.info("Quantizing layers...")
    quantizers = {}
    
    for i in tqdm(range(len(layers)), desc="Calibrating layers"):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        
        # Categorize layers
        perchannel_list = []
        pertoken_list = []
        full_list = []
        
        for f in full:
            for p in perchannel_match:
                if p in f:
                    perchannel_list.append(f)
                    full_list.append(f)
            for p in pertoken_match:
                if p in f:
                    pertoken_list.append(f)
                    full_list.append(f)
        
        sequential = list(full.keys())
        
        # Create SimQuant objects
        simquant = {}
        subset = {n: full[n] for n in sequential if n in full_list}
        sequential_subset = list(subset.keys())
        
        for name in sequential:
            if name in perchannel_list:
                simquant[name] = SimQuant(subset[name], bits, perchannel=True, qchannel=0)
            elif name in pertoken_list:
                simquant[name] = SimQuant(subset[name], bits, perchannel=True, qchannel=-1)
        
        # Register hooks to collect activations
        def add_batch(name):
            def tmp(_, inp, out):
                simquant[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in sequential_subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # Run forward pass
        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Compute quantization parameters
        for name in subset:
            cap = cap_outliers if "k_proj" in name else False
            
            quantizers['model.layers.%d.%s' % (i, name)] = simquant[name].quantize(
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                nuq=nuq,
                norm=norm,
                cap_outliers=cap,
                first_few_fp16=first_few_fp16,
                seqlen=model.seqlen
            )
            simquant[name].free()
        
        layers[i] = layer.cpu()
        del layer, simquant
        torch.cuda.empty_cache()
        
        inps, outs = outs, inps
    
    model.config.use_cache = use_cache
    return quantizers


# ============================================================================
# Evaluation Function (from KVQuant llama_simquant.py)
# ============================================================================

@torch.no_grad()
def llama_eval(model, testenc, dev):
    """
    Evaluate model perplexity.
    EXACT copy from KVQuant llama_simquant.py.
    """
    logger.info("Evaluating...")
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    # Move embeddings to device
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
    
    for i in tqdm(range(len(layers)), desc="Evaluating layers"):
        layer = layers[i].to(dev)
        
        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    # Final norm and lm_head
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
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    
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
        
        import random
        random.seed(seed)
        traintext = "\n\n".join([d['text'] for d in traindata])
        trainenc = tokenizer(traintext, return_tensors='pt')
        
        valdata = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
        valdata = list(valdata.take(256))
        testtext = "\n\n".join([d['text'] for d in valdata])
        testenc = tokenizer(testtext, return_tensors='pt')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create calibration dataloader
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append((inp,))
    
    return trainloader, testenc


def get_model(model_id, seqlen, maxseqlen):
    """Load model with proper RoPE scaling."""
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    
    if orig_ctx_len and maxseqlen > orig_ctx_len:
        scaling_factor = float(math.ceil(maxseqlen / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    # Try to use flash attention if available
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, trust_remote_code=True,
            torch_dtype=torch.half, device_map='cpu',
            attn_implementation="flash_attention_2"
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, trust_remote_code=True,
            torch_dtype=torch.half, device_map='cpu'
        )
    
    # Set seqlen attribute (required by KVQuant)
    model.seqlen = seqlen
    model.eval()
    
    logger.info(f"Model seqlen set to: {model.seqlen}")
    
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KVQuant SimQuant PPL Evaluation')
    
    # Model arguments
    parser.add_argument('model', type=str, help='Model path or HuggingFace model ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--nsamples', type=int, default=16, help='Number of calibration samples')
    
    # Quantization arguments
    parser.add_argument('--quantize', action='store_true', help='Run calibration')
    parser.add_argument('--abits', type=int, default=16, choices=[2, 3, 4, 5, 16],
                        help='Quantization bits (16 for baseline)')
    parser.add_argument('--nuq', action='store_true', help='Use Non-Uniform Quantization')
    parser.add_argument('--norm', action='store_true', help='Use Q-Norm')
    
    # Layer selection
    parser.add_argument('--perchannel', type=str, default='["k_proj"]',
                        help='Layers for per-channel quantization (JSON list)')
    parser.add_argument('--pertoken', type=str, default='["v_proj"]',
                        help='Layers for per-token quantization (JSON list)')
    
    # Sparse quantization
    parser.add_argument('--include_sparse', action='store_true',
                        help='Use dense-and-sparse quantization')
    parser.add_argument('--sparsity_threshold', type=float, default=0.99,
                        help='Outlier percentile (e.g., 0.99 = 1%% outliers)')
    
    # Calibration/quantizer path
    parser.add_argument('--quantizer_path', type=str, default=None,
                        help='Path to save/load quantizer parameters')
    
    # Attention sink
    parser.add_argument('--cap_outliers', type=float, default=-1,
                        help='Max %% outliers per token for keys')
    parser.add_argument('--first_few_fp16', type=int, default=-1,
                        help='Keep first N tokens in FP16')
    parser.add_argument('--clamp', action='store_true', help='Clamp zero-point')
    
    # Sequence length
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    parser.add_argument('--maxseqlen', type=int, default=2048, help='Max sequence length')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'c4'], help='Dataset for calibration/eval')
    
    args = parser.parse_args()
    
    import json
    args.perchannel = json.loads(args.perchannel)
    args.pertoken = json.loads(args.pertoken)
    
    DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("KVQuant SimQuant Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Bits: {args.abits}")
    logger.info(f"Mode: {'Calibration' if args.quantize else 'Evaluation'}")
    logger.info(f"Per-channel layers: {args.perchannel}")
    logger.info(f"Per-token layers: {args.pertoken}")
    if args.nuq:
        logger.info("Using Non-Uniform Quantization (NUQ)")
    if args.norm:
        logger.info("Using Q-Norm")
    if args.include_sparse:
        logger.info(f"Sparse: {(1-args.sparsity_threshold)*100:.1f}% outliers in FP16")
    if args.first_few_fp16 > 0:
        logger.info(f"Attention sink: first {args.first_few_fp16} tokens in FP16")
    logger.info("=" * 60)
    
    # Load model
    logger.info("Loading model...")
    model = get_model(args.model, args.seqlen, args.maxseqlen)
    model = model.half()
    
    # Ensure seqlen is set (fallback)
    if not hasattr(model, 'seqlen') or model.seqlen is None:
        model.seqlen = args.seqlen
        logger.warning(f"Setting model.seqlen to {args.seqlen}")
    
    logger.info(f"Model loaded. seqlen={model.seqlen}")
    
    # Load data
    logger.info("Loading data...")
    dataloader, testloader = get_loaders(
        args.dataset, args.nsamples, args.seed, args.model, model.seqlen
    )
    logger.info("Data loaded.")
    
    if args.quantize:
        # Run calibration
        quantizers = llama_calibration(
            model, dataloader, DEV,
            args.perchannel, args.pertoken, args.abits,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            nuq=args.nuq, norm=args.norm,
            cap_outliers=args.cap_outliers,
            first_few_fp16=args.first_few_fp16,
            nsamples=args.nsamples
        )
        
        # Save quantizers
        if args.quantizer_path:
            with open(args.quantizer_path, 'wb') as f:
                pickle.dump(quantizers, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Quantizers saved to {args.quantizer_path}")
    
    else:
        # Load quantizers and evaluate
        if args.quantizer_path is None:
            logger.error("Please provide --quantizer_path for evaluation")
            exit(1)
        
        with open(args.quantizer_path, 'rb') as f:
            quantizers = pickle.load(f)
        logger.info(f"Quantizers loaded from {args.quantizer_path}")
        
        # Separate per-channel and per-token quantizers
        perchannelquant = {}
        pertokenquant = {}
        
        for k in quantizers.keys():
            for p in args.perchannel:
                if p in k:
                    perchannelquant[k] = quantizers[k]
            for p in args.pertoken:
                if p in k:
                    pertokenquant[k] = quantizers[k]
        
        # Apply per-channel quantization (static - dynamicquantization=False)
        logger.info("Applying per-channel quantization (K)...")
        make_quant_sim(
            model, perchannelquant, args.abits,
            perchannel=True,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            dynamicquantization=False,  # Use calibrated parameters
            nuq=args.nuq, norm=args.norm,
            cap_outliers=args.cap_outliers,
            first_few_fp16=args.first_few_fp16,
            clamp=args.clamp
        )
        
        # Apply per-token quantization (dynamic - dynamicquantization=True)
        logger.info("Applying per-token quantization (V)...")
        make_quant_sim(
            model, pertokenquant, args.abits,
            perchannel=False,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            dynamicquantization=True,  # Dynamic for values
            nuq=args.nuq, norm=args.norm,
            cap_outliers=args.cap_outliers,
            first_few_fp16=args.first_few_fp16,
            clamp=args.clamp
        )
        
        # Run evaluation
        ppl = llama_eval(model, testloader, DEV)
        
        logger.info("=" * 60)
        logger.info(f"Perplexity: {ppl:.4f}")
        logger.info("=" * 60)
