"""
KIVI Mixed-Precision Cache V2 for Full-Rank Value Quantization

Key: 5bit per-channel quantization (uniform)
Value: 6bit/4bit mixed per-token quantization based on singular value importance

Compression equivalence:
- Original: low-rank (r) + uniform quantization
- New: full-rank (D) + 6bit/4bit mixed quantization

Split calculation (to match compression with low-rank + b_avg bit):
  b_avg * r = 6 * n_6bit + 4 * n_4bit
  n_6bit + n_4bit = D
  
  n_6bit = (b_avg * r - 4 * D) / 2
  n_4bit = D - n_6bit = (6 * D - b_avg * r) / 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from transformers.cache_utils import Cache
import math


# ============================================================================
# KIVI V2 Quantizers
# ============================================================================

class KIVIKeyQuantizerV2(nn.Module):
    """
    KIVI 5-bit per-channel quantizer for Key.
    
    Per-channel means quantization is done along the sequence dimension,
    so each channel (hidden dim) has its own scale/zero_point per group.
    """
    
    def __init__(self, n_bits: int = 5, group_size: int = 128):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.q_max = 2 ** n_bits - 1
        self.q_min = 0
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-channel quantization for Key.
        
        Args:
            x: [batch, seq_len, hidden_dim]
        
        Returns:
            Fake-quantized tensor (same shape)
        """
        if self.n_bits >= 16:
            return x
        
        # Per-channel: transpose to [batch, hidden_dim, seq_len], quantize along seq_len
        x = x.transpose(-1, -2)
        
        if self.group_size > 0 and x.shape[-1] >= self.group_size:
            *leading, seq_len = x.shape
            n_groups = seq_len // self.group_size
            
            if n_groups > 0:
                aligned = n_groups * self.group_size
                x_aligned = x[..., :aligned]
                x_rem = x[..., aligned:]
                
                x_grouped = x_aligned.view(*leading, n_groups, self.group_size)
                x_min = x_grouped.amin(dim=-1, keepdim=True)
                x_max = x_grouped.amax(dim=-1, keepdim=True)
                scale = (x_max - x_min).clamp(min=1e-5) / self.q_max
                zp = (-x_min / scale).round().clamp(self.q_min, self.q_max)
                
                x_q = (x_grouped / scale + zp).round().clamp(self.q_min, self.q_max)
                x_dq = (x_q - zp) * scale
                x_dq = x_dq.view(*leading, aligned)
                
                if x_rem.shape[-1] > 0:
                    x_dq = torch.cat([x_dq, x_rem], dim=-1)
                
                return x_dq.transpose(-1, -2)
        
        # No grouping - quantize entire sequence
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-5) / self.q_max
        zp = (-x_min / scale).round().clamp(self.q_min, self.q_max)
        x_q = (x / scale + zp).round().clamp(self.q_min, self.q_max)
        x_dq = (x_q - zp) * scale
        
        return x_dq.transpose(-1, -2)


class KIVIMixedValueQuantizerV2(nn.Module):
    """
    Mixed-precision KIVI quantizer for Value (6bit/4bit).
    
    Applies 6bit quantization to features with larger singular values,
    and 4bit quantization to features with smaller singular values.
    
    Uses KIVI-style per-token quantization (along hidden dimension).
    
    Two modes:
    1. match_compression=True: Match compression ratio with low-rank + avg_bits
       n_6bit = (avg_bits * r - 4 * D) / 2
       Only effective when r >= 2D / avg_bits
    
    2. match_compression=False: Use fixed ratio of high precision
       n_6bit = high_precision_ratio * D
    """
    
    def __init__(
        self,
        out_features: int,
        original_rank: int,
        group_size: int = 128,
        match_compression: bool = True,
        original_avg_bits: float = 5.0,  # Average bits in original low-rank scheme
        high_precision_ratio: float = 0.25,  # Used when match_compression=False
        high_bits: int = 6,
        low_bits: int = 4,
    ):
        super().__init__()
        self.out_features = out_features
        self.original_rank = original_rank
        self.group_size = group_size
        self.match_compression = match_compression
        self.original_avg_bits = original_avg_bits
        self.high_bits = high_bits
        self.low_bits = low_bits
        
        D = out_features
        r = original_rank
        
        if match_compression:
            # Mode 1: Match compression ratio
            # original_avg_bits * r = high_bits * n_high + low_bits * n_low
            # n_high + n_low = D
            # n_high = (original_avg_bits * r - low_bits * D) / (high_bits - low_bits)
            
            n_high_float = (original_avg_bits * r - self.low_bits * D) / (self.high_bits - self.low_bits)
            
            if n_high_float < 0:
                self.n_high = 0
                self.n_low = D
                self.effective_mode = "all_low"
            elif n_high_float > D:
                self.n_high = D
                self.n_low = 0
                self.effective_mode = "all_high"
            else:
                self.n_high = int(n_high_float)
                self.n_low = D - self.n_high
                self.effective_mode = "mixed"
        else:
            # Mode 2: Fixed ratio
            self.n_high = int(D * high_precision_ratio)
            self.n_low = D - self.n_high
            self.effective_mode = "fixed_ratio"
        
        # Align to group_size
        if group_size > 0 and self.n_high > 0 and self.n_high < D:
            self.n_high = (self.n_high // group_size) * group_size
            self.n_low = D - self.n_high
        
        # Calculate actual average bits
        self.avg_bits = (self.high_bits * self.n_high + self.low_bits * self.n_low) / D if D > 0 else 0
        target_bits = original_avg_bits * r / D if D > 0 else 0
        
        print(f"[MixedQuantV2] out_features={D}, original_rank={r}, mode={self.effective_mode}")
        print(f"[MixedQuantV2] n_{self.high_bits}bit={self.n_high}, n_{self.low_bits}bit={self.n_low}")
        print(f"[MixedQuantV2] Target avg bits: {target_bits:.2f}, Actual avg bits: {self.avg_bits:.2f}")
    
    @torch.no_grad()
    def _quantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        """Per-token group-wise quantization."""
        if n_bits >= 16:
            return x
        
        q_max = 2 ** n_bits - 1
        q_min = 0
        
        if self.group_size > 0 and x.shape[-1] >= self.group_size:
            *leading_dims, dim = x.shape
            n_groups = dim // self.group_size
            
            if n_groups > 0:
                aligned_dim = n_groups * self.group_size
                x_aligned = x[..., :aligned_dim]
                x_remainder = x[..., aligned_dim:]
                
                x_grouped = x_aligned.view(*leading_dims, n_groups, self.group_size)
                x_min = x_grouped.amin(dim=-1, keepdim=True)
                x_max = x_grouped.amax(dim=-1, keepdim=True)
                scale = (x_max - x_min).clamp(min=1e-5) / q_max
                zero_point = (-x_min / scale).round().clamp(q_min, q_max)
                
                x_quant = (x_grouped / scale + zero_point).round().clamp(q_min, q_max)
                x_dequant = (x_quant - zero_point) * scale
                x_dequant = x_dequant.view(*leading_dims, aligned_dim)
                
                if x_remainder.shape[-1] > 0:
                    x_dequant = torch.cat([x_dequant, x_remainder], dim=-1)
                
                return x_dequant
        
        # No grouping
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-5) / q_max
        zero_point = (-x_min / scale).round().clamp(q_min, q_max)
        x_quant = (x / scale + zero_point).round().clamp(q_min, q_max)
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply mixed-precision quantization (6bit/4bit).
        
        Features are assumed to be ordered by singular value importance
        (largest first), so first n_high features get high_bits, rest get low_bits.
        """
        if self.n_high == self.out_features:
            return self._quantize(x, self.high_bits)
        elif self.n_high == 0:
            return self._quantize(x, self.low_bits)
        else:
            x_high = x[..., :self.n_high]
            x_low = x[..., self.n_high:]
            
            x_high_quant = self._quantize(x_high, self.high_bits)
            x_low_quant = self._quantize(x_low, self.low_bits)
            
            return torch.cat([x_high_quant, x_low_quant], dim=-1)


# ============================================================================
# Full-Rank Value Linear with 6bit/4bit Mixed KIVI Quantization
# ============================================================================

class ALRDLinear_KIVI_Value_FullRank_MixedV2(nn.Module):
    """
    Full-rank Value linear layer with 6bit/4bit mixed-precision KIVI quantization.
    
    - BLinear: in_features -> out_features (full rank, no truncation)
    - ALinear: out_features -> out_features
    - Quantization: 6bit for important features, 4bit for others
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        original_rank: int,
        bias: bool = True,
        group_size: int = 128,
        residual_length: int = 128,
        match_compression: bool = True,
        original_avg_bits: float = 5.0,
        high_precision_ratio: float = 0.25,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.original_rank = original_rank
        self.group_size = group_size
        self.residual_length = residual_length
        
        # Full-rank projection
        self.BLinear = nn.Linear(in_features, out_features, bias=False)
        self.ALinear = nn.Linear(out_features, out_features, bias=bias)
        
        # Mixed-precision quantizer (6bit/4bit)
        self.mixed_quantizer = KIVIMixedValueQuantizerV2(
            out_features=out_features,
            original_rank=original_rank,
            group_size=group_size,
            match_compression=match_compression,
            original_avg_bits=original_avg_bits,
            high_precision_ratio=high_precision_ratio,
            high_bits=6,
            low_bits=4,
        )
        
        self.n_high = self.mixed_quantizer.n_high
        self.n_low = self.mixed_quantizer.n_low
        self.rank = out_features
        self.rank_lists = self._split_rank_for_hada(out_features)
    
    def _split_rank_for_hada(self, rank):
        """Split rank into chunks suitable for Hadamard transform."""
        def is_pow2(n):
            return (n & (n - 1) == 0) and (n > 0)
        
        hada_list = []
        rank_lists = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        res_rank = rank
        for item in rank_lists:
            if item == 1 or res_rank == 1 or res_rank == 0:
                break
            if is_pow2(res_rank):
                hada_list.append(res_rank)
                break
            while res_rank >= item:
                times = res_rank // item
                if is_pow2(times):
                    hada_list.append(times * item)
                    res_rank = res_rank % item
                else:
                    hada_list.append(item)
                    res_rank = res_rank - item
        return hada_list
    
    def fuse_hadamard(self):
        """Apply Hadamard transform to weights."""
        def hadamard_transform(x):
            n = x.size(1)
            if n & (n - 1) != 0:
                raise ValueError("Input size must be a power of 2.")
            H = torch.tensor([[1, 1], [1, -1]], dtype=x.dtype).to(x.device)
            for i in range(1, int(n.bit_length()-1)):
                H = torch.kron(H, torch.tensor([[1, 1], [1, -1]], dtype=x.dtype).to(H.device))
            return torch.matmul(x, H) / torch.tensor(n, dtype=x.dtype).sqrt()
        
        VT_weight = self.BLinear.weight.data
        U_weight = self.ALinear.weight.data
        total_rank = 0
        
        for rank in self.rank_lists:
            VT_chunk = VT_weight[total_rank:total_rank + rank, :].contiguous()
            VT_chunk = VT_chunk.transpose(0, 1).contiguous()
            VT_chunk = VT_chunk.view(-1, VT_chunk.shape[-1]).contiguous()
            VT_chunk = hadamard_transform(VT_chunk)
            self.BLinear.weight.data[total_rank:total_rank + rank, :] = VT_chunk.t()
            
            U_chunk = U_weight[:, total_rank:total_rank + rank].contiguous()
            U_chunk = U_chunk.view(-1, U_chunk.shape[-1]).contiguous()
            U_chunk = hadamard_transform(U_chunk)
            self.ALinear.weight.data[:, total_rank:total_rank + rank] = U_chunk.view_as(
                self.ALinear.weight.data[:, total_rank:total_rank + rank]
            )
            
            total_rank += rank
    
    def quantize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply mixed-precision quantization."""
        return self.mixed_quantizer(latents)
    
    def quantize_latent_mixed(self, latents: torch.Tensor) -> torch.Tensor:
        """Mixed-precision quantization with residual support."""
        seq_len = latents.shape[1]
        
        if seq_len <= self.residual_length:
            return latents
        
        n_quant = seq_len - self.residual_length
        if self.group_size > 0:
            n_quant = (n_quant // self.group_size) * self.group_size
        
        if n_quant <= 0:
            return latents
        
        latents_to_quant = latents[:, :n_quant, :]
        latents_residual = latents[:, n_quant:, :]
        
        latents_quantized = self.quantize_latent(latents_to_quant)
        
        return torch.cat([latents_quantized, latents_residual], dim=1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.BLinear(input)
        y = self.quantize_latent(y)
        return self.ALinear(y)


# ============================================================================
# KIVI Mixed-Precision Cache V2 (Key:5bit, Value:6bit/4bit)
# ============================================================================

class KIVIMixedPrecisionCacheV2(Cache):
    """
    KIVI cache V2 with:
    - Key: 5bit per-channel quantization (uniform)
    - Value: 6bit/4bit mixed per-token quantization
    
    Supports per-layer different ranks for Value quantization.
    """
    
    def __init__(
        self,
        k_bits: int = 5,  # Key uses 5bit
        out_features: int = 4096,
        layer_original_ranks: Dict[int, int] = None,
        default_original_rank: int = 256,
        group_size: int = 128,
        residual_length: int = 128,
        # Value mixed precision options
        match_compression: bool = True,
        original_avg_bits: float = 5.0,  # Original scheme's average bits
        high_precision_ratio: float = 0.25,
        high_bits: int = 6,  # Value high precision
        low_bits: int = 4,   # Value low precision
    ):
        super().__init__()
        self.k_bits = k_bits
        self.out_features = out_features
        self.layer_original_ranks = layer_original_ranks or {}
        self.default_original_rank = default_original_rank
        self.group_size = group_size
        self.residual_length = residual_length
        
        # Value mixed precision options
        self.match_compression = match_compression
        self.original_avg_bits = original_avg_bits
        self.high_precision_ratio = high_precision_ratio
        self.high_bits = high_bits
        self.low_bits = low_bits
        
        # Key quantizer: 5bit per-channel
        self.key_quantizer = KIVIKeyQuantizerV2(n_bits=k_bits, group_size=group_size)
        
        # Per-layer Value quantizers (created lazily)
        self._value_quantizers: Dict[int, KIVIMixedValueQuantizerV2] = {}
        
        # Cache storage: (key_quant, key_residual, value_quant, value_residual)
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._seen_tokens = 0
        
        print(f"[KIVICacheV2] Key: {k_bits}bit, Value: {high_bits}bit/{low_bits}bit mixed")
    
    def _get_value_quantizer(self, layer_idx: int) -> KIVIMixedValueQuantizerV2:
        """Get or create a value quantizer for the given layer."""
        if layer_idx not in self._value_quantizers:
            original_rank = self.layer_original_ranks.get(layer_idx, self.default_original_rank)
            
            self._value_quantizers[layer_idx] = KIVIMixedValueQuantizerV2(
                out_features=self.out_features,
                original_rank=original_rank,
                group_size=self.group_size,
                match_compression=self.match_compression,
                original_avg_bits=self.original_avg_bits,
                high_precision_ratio=self.high_precision_ratio,
                high_bits=self.high_bits,
                low_bits=self.low_bits,
            )
        
        return self._value_quantizers[layer_idx]
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __iter__(self):
        for layer_idx in sorted(self._cache.keys()):
            yield self._cache[layer_idx]
    
    def __getitem__(self, layer_idx: int):
        if layer_idx not in self._cache:
            return None
        k_q, k_r, v_q, v_r = self._cache[layer_idx]
        all_k = torch.cat([k_q, k_r], dim=1) if k_q is not None else k_r
        all_v = torch.cat([v_q, v_r], dim=1) if v_q is not None else v_r
        return (all_k, all_v)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self._cache:
            return 0
        k_q, k_r, _, _ = self._cache[layer_idx]
        quant_len = k_q.shape[1] if k_q is not None else 0
        res_len = k_r.shape[1] if k_r is not None else 0
        return quant_len + res_len
    
    def get_usable_length(self, new_seq_len: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)
    
    def get_max_length(self) -> Optional[int]:
        return None
    
    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens if self._seen_tokens > 0 else self.get_seq_length(0)
    
    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with 5bit Key and 6bit/4bit mixed Value quantization."""
        
        # Handle both 3D [batch, seq, dim] and 4D [batch, heads, seq, head_dim]
        if key_states.dim() == 4:
            b, h, s, d = key_states.shape
            key_flat = key_states.transpose(1, 2).reshape(b, s, h * d)
            value_flat = value_states.transpose(1, 2).reshape(b, s, h * d)
            is_4d = True
        else:
            key_flat = key_states
            value_flat = value_states
            is_4d = False
            b, s, _ = key_flat.shape
        
        value_quantizer = self._get_value_quantizer(layer_idx)
        
        if layer_idx not in self._cache:
            # First update
            self._seen_tokens = key_flat.shape[1]
            
            if key_flat.shape[1] <= self.residual_length:
                self._cache[layer_idx] = (None, key_flat.clone(), None, value_flat.clone())
                return key_states, value_states
            
            n_quant = key_flat.shape[1] - self.residual_length
            if self.group_size > 0:
                n_quant = (n_quant // self.group_size) * self.group_size
            
            if n_quant > 0:
                key_to_quant = key_flat[:, :n_quant, :]
                key_res = key_flat[:, n_quant:, :].clone()
                value_to_quant = value_flat[:, :n_quant, :]
                value_res = value_flat[:, n_quant:, :].clone()
                
                # Key: 5bit per-channel
                key_quant = self.key_quantizer(key_to_quant)
                # Value: 6bit/4bit mixed per-token
                value_quant = value_quantizer(value_to_quant)
                
                self._cache[layer_idx] = (key_quant, key_res, value_quant, value_res)
                
                all_keys = torch.cat([key_quant, key_res], dim=1)
                all_values = torch.cat([value_quant, value_res], dim=1)
            else:
                self._cache[layer_idx] = (None, key_flat.clone(), None, value_flat.clone())
                all_keys = key_flat
                all_values = value_flat
        else:
            # Append to existing
            key_quant, key_res, value_quant, value_res = self._cache[layer_idx]
            self._seen_tokens += key_flat.shape[1]
            
            if key_res is not None:
                key_res = torch.cat([key_res, key_flat], dim=1)
                value_res = torch.cat([value_res, value_flat], dim=1)
            else:
                key_res = key_flat.clone()
                value_res = value_flat.clone()
            
            # Check if need to move to quantized
            if key_res.shape[1] > self.residual_length:
                n_to_move = key_res.shape[1] - self.residual_length
                if self.group_size > 0:
                    n_to_move = (n_to_move // self.group_size) * self.group_size
                
                if n_to_move > 0:
                    key_to_move = key_res[:, :n_to_move, :]
                    value_to_move = value_res[:, :n_to_move, :]
                    
                    key_move_quant = self.key_quantizer(key_to_move)
                    value_move_quant = value_quantizer(value_to_move)
                    
                    if key_quant is not None:
                        key_quant = torch.cat([key_quant, key_move_quant], dim=1)
                        value_quant = torch.cat([value_quant, value_move_quant], dim=1)
                    else:
                        key_quant = key_move_quant
                        value_quant = value_move_quant
                    
                    key_res = key_res[:, n_to_move:, :]
                    value_res = value_res[:, n_to_move:, :]
            
            self._cache[layer_idx] = (key_quant, key_res, value_quant, value_res)
            
            if key_quant is not None:
                all_keys = torch.cat([key_quant, key_res], dim=1)
                all_values = torch.cat([value_quant, value_res], dim=1)
            else:
                all_keys = key_res
                all_values = value_res
        
        # Convert back to 4D if needed
        if is_4d:
            all_keys = all_keys.view(b, -1, h, d).transpose(1, 2)
            all_values = all_values.view(b, -1, h, d).transpose(1, 2)
        
        return all_keys, all_values
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search."""
        for layer_idx in self._cache:
            k_q, k_r, v_q, v_r = self._cache[layer_idx]
            if k_q is not None:
                k_q = k_q.index_select(0, beam_idx)
                v_q = v_q.index_select(0, beam_idx)
            if k_r is not None:
                k_r = k_r.index_select(0, beam_idx)
                v_r = v_r.index_select(0, beam_idx)
            self._cache[layer_idx] = (k_q, k_r, v_q, v_r)
    
    def reset(self):
        """Reset cache."""
        self._cache.clear()
        self._value_quantizers.clear()
        self._seen_tokens = 0
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information for debugging."""
        if not self._cache:
            return {"status": "empty"}
        
        layer_0 = self._cache.get(0)
        if layer_0 is None:
            return {"status": "no layer 0"}
        
        k_q, k_r, v_q, v_r = layer_0
        
        per_layer_info = {}
        for layer_idx, quantizer in self._value_quantizers.items():
            per_layer_info[layer_idx] = {
                "original_rank": quantizer.original_rank,
                "n_high": quantizer.n_high,
                "n_low": quantizer.n_low,
                "avg_bits": quantizer.avg_bits,
                "mode": quantizer.effective_mode,
            }
        
        return {
            "num_layers": len(self._cache),
            "total_seq_len": self.get_seq_length(0),
            "quantized_len": k_q.shape[1] if k_q is not None else 0,
            "residual_len": k_r.shape[1] if k_r is not None else 0,
            "k_bits": self.k_bits,
            "v_high_bits": self.high_bits,
            "v_low_bits": self.low_bits,
            "per_layer_value_quant": per_layer_info,
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_mixed_precision_cache_v2(
    k_bits: int = 5,
    out_features: int = 4096,
    layer_original_ranks: Dict[int, int] = None,
    default_original_rank: int = 256,
    group_size: int = 128,
    residual_length: int = 128,
    match_compression: bool = True,
    original_avg_bits: float = 5.0,
    high_precision_ratio: float = 0.25,
    high_bits: int = 6,
    low_bits: int = 4,
) -> KIVIMixedPrecisionCacheV2:
    """
    Create a mixed-precision KIVI cache V2.
    
    Key: k_bits (default 5bit) per-channel quantization
    Value: high_bits/low_bits (default 6bit/4bit) mixed per-token quantization
    
    Args:
        k_bits: Number of bits for Key quantization
        out_features: Output dimension D
        layer_original_ranks: Dict mapping layer_idx -> original_rank
        default_original_rank: Fallback rank
        group_size: Group size for quantization
        residual_length: Recent tokens in full precision
        match_compression: If True, match compression with low-rank + original_avg_bits
        original_avg_bits: Average bits in original low-rank scheme
        high_precision_ratio: When match_compression=False, ratio of high precision features
        high_bits: High precision bits for Value (default 6)
        low_bits: Low precision bits for Value (default 4)
    
    Returns:
        KIVIMixedPrecisionCacheV2
    """
    return KIVIMixedPrecisionCacheV2(
        k_bits=k_bits,
        out_features=out_features,
        layer_original_ranks=layer_original_ranks,
        default_original_rank=default_original_rank,
        group_size=group_size,
        residual_length=residual_length,
        match_compression=match_compression,
        original_avg_bits=original_avg_bits,
        high_precision_ratio=high_precision_ratio,
        high_bits=high_bits,
        low_bits=low_bits,
    )


def calculate_mixed_precision_split_v2(
    out_features: int,
    original_rank: int,
    original_avg_bits: float = 5.0,
    high_bits: int = 6,
    low_bits: int = 4,
) -> Tuple[int, int]:
    """
    Calculate the high/low bit split for V2 mixed-precision quantization.
    
    Args:
        out_features: Full dimension (D)
        original_rank: Rank that would be used in low-rank scheme (r)
        original_avg_bits: Average bits per element in original scheme
        high_bits: High precision bits
        low_bits: Low precision bits
    
    Returns:
        (n_high, n_low): Number of features for each precision
    """
    D = out_features
    r = original_rank
    
    # original_avg_bits * r = high_bits * n_high + low_bits * n_low
    # n_high + n_low = D
    # n_high = (original_avg_bits * r - low_bits * D) / (high_bits - low_bits)
    
    n_high = (original_avg_bits * r - low_bits * D) / (high_bits - low_bits)
    
    if n_high < 0:
        return 0, D
    elif n_high > D:
        return D, 0
    else:
        n_high = int(n_high)
        return n_high, D - n_high
