import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, Optional

@torch.no_grad()
def quantize_tensor(w: torch.tensor, n_bits, group_size, sym, clip_ratio=1.0) -> torch.tensor:
    savedShape = w.shape
    assert w.dim() == 2 


    if group_size > 0:
        assert w.shape[-1] % group_size == 0
        w = w.reshape(-1, group_size) # row-major order

    assert w.dim() == 2, "Weight format: [-1, group]"
    #assert n_bits < 16
    if n_bits >= 16:
        return w

    if sym:
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    else:
        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)

    if sym:
        q_max = (2**(n_bits-1)-1)
        q_min = (-2**(n_bits-1))
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        base = torch.zeros_like(scales)
    else:
        q_max = (2**(n_bits)-1)
        q_min = (0)
        if clip_ratio < 1.0:
            w_max *= clip_ratio
            w_min *= clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
    w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    
    return w.reshape(savedShape)




class Quantizer(nn.Module):
    def __init__(self,
            n_bits: int, 
            group_size: int, 
            sym: bool,
            clip_ratio: float     
        ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.sym = sym
        self.clip_ratio = clip_ratio
        

    @torch.no_grad()
    def forward(self, x):
        if self.n_bits >= 16:
            return x 
        
        qFunction = partial(
            quantize_tensor, 
            n_bits=self.n_bits,
            group_size=self.group_size,
            sym=self.sym,
            clip_ratio=self.clip_ratio
        )

        savedShape = x.shape
        x = x.view(-1, savedShape[-1])
        assert self.group_size == 0 or (savedShape[-1]) % self.group_size == 0, "Group size should be divisible by (dim)."

        x = qFunction(x)
        
        return x.view(savedShape)
        
    def to(self, *args, **kwargs):
        super(Quantizer, self).to(*args, **kwargs)
        return self


@torch.no_grad()
def quantize_tensor_forward(w: torch.tensor, n_bits, group_size, sym=True, clip_ratio=1.0):
    savedShape = w.shape
    assert w.dim() == 2 


    if group_size > 0:
        assert w.shape[-1] % group_size == 0
        w = w.reshape(-1, group_size) # row-major order

    assert w.dim() == 2, "Weight format: [-1, group]"
    #assert n_bits < 16
    if n_bits >= 16:
        return w

    if sym:
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    else:
        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)

    if sym:
        q_max = (2**(n_bits-1)-1)
        q_min = (-2**(n_bits-1))
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        base = torch.zeros_like(scales)
    else:
        q_max = (2**(n_bits)-1)
        q_min = (0)
        if clip_ratio < 1.0:
            w_max *= clip_ratio
            w_min *= clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
    #w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    w = torch.clamp(torch.round(w / scales) + base, q_min, q_max)
    
    scales = scales[:,0].reshape(-1)

    return w.reshape(savedShape), scales



class Quantizer2(nn.Module):
    def __init__(self,
            n_bits: int, 
            group_size: int, 
            sym: bool,
            clip_ratio: float     
        ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.sym = True
        self.clip_ratio = clip_ratio
    
    @torch.no_grad()
    def forward(self, x):
        if self.n_bits >= 16:
            return x 
        
        qFunction = partial(
            quantize_tensor_forward, 
            n_bits=self.n_bits,
            group_size=self.group_size,
            sym=self.sym,
            clip_ratio=self.clip_ratio
        )

        savedShape = x.shape
        x = x.view(-1, savedShape[-1])
        assert self.group_size == 0 or (savedShape[-1]) % self.group_size == 0, "Group size should be divisible by (dim)."

        x, scales = qFunction(x)
        
        return x.view(savedShape), scales
        
    def to(self, *args, **kwargs):
        super(Quantizer2, self).to(*args, **kwargs)
        return self


# ============================================================================
# KIVI-style Quantization Functions for Low-Rank KV Cache
# ============================================================================

@torch.no_grad()
def kivi_quantize_per_channel(
    x: torch.Tensor, 
    n_bits: int = 2, 
    group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    KIVI-style per-channel quantization (for Key cache).
    Quantizes along the token/sequence dimension.
    
    Args:
        x: Input tensor of shape [batch, seq_len, dim] or [batch, heads, seq_len, head_dim]
        n_bits: Number of quantization bits
        group_size: Group size for quantization (0 means no grouping)
        
    Returns:
        x_quant: Fake-quantized tensor (same shape as input)
        scale: Quantization scales
        zero_point: Zero points
    """
    if n_bits >= 16:
        return x, None, None
    
    original_shape = x.shape
    q_max = 2 ** n_bits - 1
    q_min = 0
    
    # Transpose to make channel (dim) the quantization axis
    # [batch, seq_len, dim] -> [batch, dim, seq_len]
    x = x.transpose(-1, -2)
    
    if group_size > 0 and x.shape[-1] >= group_size:
        # Group-wise quantization along seq_len dimension
        *leading_dims, seq_len = x.shape
        n_groups = (seq_len + group_size - 1) // group_size
        padded_len = n_groups * group_size
        
        if padded_len != seq_len:
            padding = padded_len - seq_len
            x = F.pad(x, (0, padding), value=0)
        
        x = x.view(*leading_dims, n_groups, group_size)
        
        # Min-max per group
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-5) / q_max
        zero_point = (-x_min / scale).round().clamp(q_min, q_max)
        
        # Quantize and dequantize
        x_quant = (x / scale + zero_point).round().clamp(q_min, q_max)
        x_dequant = (x_quant - zero_point) * scale
        
        # Reshape back
        x_dequant = x_dequant.view(*leading_dims, padded_len)
        if padded_len != seq_len:
            x_dequant = x_dequant[..., :seq_len]
        
        scale = scale.squeeze(-1)
        zero_point = zero_point.squeeze(-1)
    else:
        # Quantize entire sequence
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-5) / q_max
        zero_point = (-x_min / scale).round().clamp(q_min, q_max)
        
        x_quant = (x / scale + zero_point).round().clamp(q_min, q_max)
        x_dequant = (x_quant - zero_point) * scale
    
    # Transpose back: [batch, dim, seq_len] -> [batch, seq_len, dim]
    x_dequant = x_dequant.transpose(-1, -2)
    
    return x_dequant, scale, zero_point


@torch.no_grad()
def kivi_quantize_per_token(
    x: torch.Tensor, 
    n_bits: int = 2, 
    group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    KIVI-style per-token quantization (for Value cache).
    Quantizes along the hidden/feature dimension.
    
    Args:
        x: Input tensor of shape [batch, seq_len, dim] or [batch, heads, seq_len, head_dim]
        n_bits: Number of quantization bits
        group_size: Group size for quantization (0 means no grouping)
        
    Returns:
        x_quant: Fake-quantized tensor (same shape as input)
        scale: Quantization scales
        zero_point: Zero points
    """
    if n_bits >= 16:
        return x, None, None
    
    original_shape = x.shape
    q_max = 2 ** n_bits - 1
    q_min = 0
    
    if group_size > 0 and x.shape[-1] >= group_size:
        # Group-wise quantization along dim
        *leading_dims, dim = x.shape
        n_groups = (dim + group_size - 1) // group_size
        padded_dim = n_groups * group_size
        
        if padded_dim != dim:
            padding = padded_dim - dim
            x = F.pad(x, (0, padding), value=0)
        
        x = x.view(*leading_dims, n_groups, group_size)
        
        # Min-max per group
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-5) / q_max
        zero_point = (-x_min / scale).round().clamp(q_min, q_max)
        
        # Quantize and dequantize
        x_quant = (x / scale + zero_point).round().clamp(q_min, q_max)
        x_dequant = (x_quant - zero_point) * scale
        
        # Reshape back
        x_dequant = x_dequant.view(*leading_dims, padded_dim)
        if padded_dim != dim:
            x_dequant = x_dequant[..., :dim]
        
        scale = scale.squeeze(-1)
        zero_point = zero_point.squeeze(-1)
    else:
        # Quantize entire dim
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-5) / q_max
        zero_point = (-x_min / scale).round().clamp(q_min, q_max)
        
        x_quant = (x / scale + zero_point).round().clamp(q_min, q_max)
        x_dequant = (x_quant - zero_point) * scale
    
    return x_dequant, scale, zero_point


class KIVIKeyQuantizer(nn.Module):
    """KIVI-style per-channel quantizer for Key cache (low-rank latent)."""
    
    def __init__(
        self,
        n_bits: int = 2,
        group_size: int = 128,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fake quantization for Key latent.
        
        Args:
            x: [batch, seq_len, rank] - output of k_proj.BLinear
        """
        x_quant, _, _ = kivi_quantize_per_channel(x, self.n_bits, self.group_size)
        return x_quant


class KIVIValueQuantizer(nn.Module):
    """KIVI-style per-token quantizer for Value cache (low-rank latent)."""
    
    def __init__(
        self,
        n_bits: int = 2,
        group_size: int = 128,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fake quantization for Value latent.
        
        Args:
            x: [batch, seq_len, rank] - output of v_proj.BLinear
        """
        x_quant, _, _ = kivi_quantize_per_token(x, self.n_bits, self.group_size)
        return x_quant


class KIVIMixedQuantizer(nn.Module):
    """
    Mixed precision quantizer following KIVI design:
    - Recent tokens (residual_length) kept in full precision
    - Older tokens quantized
    """
    
    def __init__(
        self,
        n_bits: int = 2,
        group_size: int = 128,
        residual_length: int = 32,
        per_channel: bool = False,  # True for Key, False for Value
    ):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.per_channel = per_channel
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize with residual: keep recent tokens in full precision.
        
        Args:
            x: [batch, seq_len, dim]
        """
        if self.n_bits >= 16:
            return x
        
        seq_len = x.shape[1]
        
        if seq_len <= self.residual_length:
            # All tokens fit in residual, no quantization
            return x
        
        # Split into quantized and residual parts
        n_quant = seq_len - self.residual_length
        # Align to group_size
        if self.group_size > 0:
            n_quant = (n_quant // self.group_size) * self.group_size
        
        if n_quant <= 0:
            return x
        
        x_to_quant = x[:, :n_quant, :]
        x_residual = x[:, n_quant:, :]
        
        if self.per_channel:
            x_quantized, _, _ = kivi_quantize_per_channel(
                x_to_quant, self.n_bits, self.group_size
            )
        else:
            x_quantized, _, _ = kivi_quantize_per_token(
                x_to_quant, self.n_bits, self.group_size
            )
        
        return torch.cat([x_quantized, x_residual], dim=1)