"""
KIVI Mixed-Precision Cache for Full-Rank Value Quantization

This module implements a mixed-precision quantization scheme:
- Original: low-rank (r) + 3bit uniform quantization
- New: full-rank (D) + 4bit/2bit mixed quantization

The compression ratio is kept the same:
  3r / (16D) = (4 * n_4bit + 2 * n_2bit) / (16D)
  
Where:
  n_4bit = (3r - 2D) / 2  (features with larger singular values)
  n_2bit = (4D - 3r) / 2  (features with smaller singular values)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from transformers.cache_utils import Cache
import math


# ============================================================================
# KIVI Mixed-Precision Quantizer
# ============================================================================

class KIVIMixedPrecisionQuantizer(nn.Module):
    """
    Mixed-precision KIVI quantizer for full-rank Value.
    
    Applies 4bit quantization to features with larger singular values,
    and 2bit quantization to features with smaller singular values.
    
    Uses KIVI-style per-token quantization (along hidden dimension).
    """
    
    def __init__(
        self,
        out_features: int,
        original_rank: int,
        group_size: int = 128,
    ):
        super().__init__()
        self.out_features = out_features
        self.original_rank = original_rank
        self.group_size = group_size
        
        # Calculate n_4bit and n_2bit to match compression ratio
        # Original: rank * 3bit
        # New: n_4bit * 4bit + n_2bit * 2bit
        # Constraint: 3r = 4 * n_4bit + 2 * n_2bit, n_4bit + n_2bit = D
        
        D = out_features
        r = original_rank
        
        # n_4bit = (3r - 2D) / 2
        n_4bit_float = (3 * r - 2 * D) / 2
        
        if n_4bit_float < 0:
            # Compression too high, need even lower precision
            # Use all 2bit
            self.n_4bit = 0
            self.n_2bit = D
            print(f"[MixedQuant] Warning: rank={r} too small for 4bit+2bit, using all 2bit")
        elif n_4bit_float > D:
            # Compression too low, use all 4bit
            self.n_4bit = D
            self.n_2bit = 0
            print(f"[MixedQuant] Warning: rank={r} too large, using all 4bit")
        else:
            self.n_4bit = int(n_4bit_float)
            self.n_2bit = D - self.n_4bit
        
        # Align to group_size for efficient quantization
        if group_size > 0 and self.n_4bit > 0 and self.n_4bit < D:
            self.n_4bit = (self.n_4bit // group_size) * group_size
            self.n_2bit = D - self.n_4bit
        
        # Calculate actual compression ratio
        actual_bits = (4 * self.n_4bit + 2 * self.n_2bit) / D
        original_bits = 3 * r / D
        
        print(f"[MixedQuant] out_features={D}, original_rank={r}")
        print(f"[MixedQuant] n_4bit={self.n_4bit}, n_2bit={self.n_2bit}")
        print(f"[MixedQuant] Original avg bits: {original_bits:.2f}, Actual avg bits: {actual_bits:.2f}")
    
    @torch.no_grad()
    def quantize_4bit(self, x: torch.Tensor) -> torch.Tensor:
        """KIVI-style 4bit per-token quantization."""
        return self._quantize(x, n_bits=4)
    
    @torch.no_grad()
    def quantize_2bit(self, x: torch.Tensor) -> torch.Tensor:
        """KIVI-style 2bit per-token quantization."""
        return self._quantize(x, n_bits=2)
    
    @torch.no_grad()
    def _quantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        """
        KIVI-style per-token quantization.
        
        Args:
            x: [batch, seq_len, dim]
            n_bits: quantization bits
        
        Returns:
            Fake-quantized tensor (same shape)
        """
        if n_bits >= 16:
            return x
        
        q_max = 2 ** n_bits - 1
        q_min = 0
        
        original_shape = x.shape
        
        if self.group_size > 0 and x.shape[-1] >= self.group_size:
            # Group-wise quantization along last dim
            *leading_dims, dim = x.shape
            n_groups = dim // self.group_size
            
            if n_groups > 0:
                aligned_dim = n_groups * self.group_size
                x_aligned = x[..., :aligned_dim]
                x_remainder = x[..., aligned_dim:]
                
                # Reshape to groups
                x_grouped = x_aligned.view(*leading_dims, n_groups, self.group_size)
                
                # Per-group min-max
                x_min = x_grouped.amin(dim=-1, keepdim=True)
                x_max = x_grouped.amax(dim=-1, keepdim=True)
                scale = (x_max - x_min).clamp(min=1e-5) / q_max
                zero_point = (-x_min / scale).round().clamp(q_min, q_max)
                
                # Quantize and dequantize
                x_quant = (x_grouped / scale + zero_point).round().clamp(q_min, q_max)
                x_dequant = (x_quant - zero_point) * scale
                
                # Reshape back
                x_dequant = x_dequant.view(*leading_dims, aligned_dim)
                
                # Handle remainder (keep as-is)
                if x_remainder.shape[-1] > 0:
                    x_dequant = torch.cat([x_dequant, x_remainder], dim=-1)
                
                return x_dequant
        
        # No grouping - quantize entire dim
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
        Apply mixed-precision quantization.
        
        Features are assumed to be ordered by singular value importance
        (largest first), so first n_4bit features get 4bit, rest get 2bit.
        
        Args:
            x: [batch, seq_len, out_features]
        
        Returns:
            Mixed-precision quantized tensor
        """
        if self.n_4bit == self.out_features:
            # All 4bit
            return self.quantize_4bit(x)
        elif self.n_4bit == 0:
            # All 2bit
            return self.quantize_2bit(x)
        else:
            # Mixed: first n_4bit features -> 4bit, rest -> 2bit
            x_4bit = x[..., :self.n_4bit]
            x_2bit = x[..., self.n_4bit:]
            
            x_4bit_quant = self.quantize_4bit(x_4bit)
            x_2bit_quant = self.quantize_2bit(x_2bit)
            
            return torch.cat([x_4bit_quant, x_2bit_quant], dim=-1)


# ============================================================================
# Full-Rank Value Linear with Mixed-Precision KIVI Quantization
# ============================================================================

class ALRDLinear_KIVI_Value_FullRank_Mixed(nn.Module):
    """
    Full-rank Value linear layer with mixed-precision KIVI quantization.
    
    Instead of truncating to rank r and using 3bit uniform quantization,
    this keeps full rank D and uses 4bit/2bit mixed quantization to achieve
    the same compression ratio.
    
    Key insight: Features corresponding to larger singular values (more important)
    get 4bit quantization, while smaller ones get 2bit.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension (D)
        original_rank: The rank that would have been used with 3bit quantization (r)
        bias: Whether to use bias
        group_size: Group size for KIVI quantization
        residual_length: Number of recent tokens to keep in full precision
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        original_rank: int,
        bias: bool = True,
        group_size: int = 128,
        residual_length: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.original_rank = original_rank
        self.group_size = group_size
        self.residual_length = residual_length
        
        # Full-rank projection (no truncation)
        self.BLinear = nn.Linear(in_features, out_features, bias=False)
        self.ALinear = nn.Linear(out_features, out_features, bias=bias)
        
        # Mixed-precision quantizer
        self.mixed_quantizer = KIVIMixedPrecisionQuantizer(
            out_features=out_features,
            original_rank=original_rank,
            group_size=group_size,
        )
        
        # Store for reference
        self.n_4bit = self.mixed_quantizer.n_4bit
        self.n_2bit = self.mixed_quantizer.n_2bit
        
        # For Hadamard transform compatibility
        self.rank = out_features  # Full rank
        self.rank_lists = self._split_rank_for_hada(out_features)
    
    def _split_rank_for_hada(self, rank):
        """Split rank into chunks suitable for Hadamard transform (power of 2)."""
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
        """Apply Hadamard transform to BLinear and ALinear weights for better quantization."""
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
        
        print(f"ALRDLinear_KIVI_Value_FullRank_Mixed fuse_hadamard: out_features={self.out_features}, rank_lists={self.rank_lists}")
        
        for rank in self.rank_lists:
            # Transform BLinear (VT)
            VT_chunk = VT_weight[total_rank:total_rank + rank, :].contiguous()
            VT_chunk = VT_chunk.transpose(0, 1).contiguous()
            VT_chunk = VT_chunk.view(-1, VT_chunk.shape[-1]).contiguous()
            VT_chunk = hadamard_transform(VT_chunk)
            self.BLinear.weight.data[total_rank:total_rank + rank, :] = VT_chunk.t()
            
            # Transform ALinear (U)
            U_chunk = U_weight[:, total_rank:total_rank + rank].contiguous()
            U_chunk = U_chunk.view(-1, U_chunk.shape[-1]).contiguous()
            U_chunk = hadamard_transform(U_chunk)
            self.ALinear.weight.data[:, total_rank:total_rank + rank] = U_chunk.view_as(
                self.ALinear.weight.data[:, total_rank:total_rank + rank]
            )
            
            total_rank += rank
    
    def quantize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Apply mixed-precision quantization to latents.
        
        Args:
            latents: [batch, seq_len, out_features]
        
        Returns:
            Mixed-precision quantized latents
        """
        return self.mixed_quantizer(latents)
    
    def quantize_latent_mixed(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Mixed-precision quantization with residual support.
        Keep recent tokens in full precision.
        """
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
# Mixed-Precision KIVI Cache
# ============================================================================

class KIVIMixedPrecisionCache(Cache):
    """
    KIVI cache with mixed-precision quantization for Value.
    
    Key: Uses standard KIVI per-channel quantization (k_bits)
    Value: Uses mixed 4bit/2bit quantization based on singular value importance
    
    Supports per-layer different ranks for Value quantization.
    
    Compatible with model.generate()
    """
    
    def __init__(
        self,
        k_bits: int = 2,
        out_features: int = 4096,
        layer_original_ranks: Dict[int, int] = None,  # Per-layer ranks
        default_original_rank: int = 256,  # Fallback if layer not in dict
        group_size: int = 128,
        residual_length: int = 128,
    ):
        super().__init__()
        self.k_bits = k_bits
        self.out_features = out_features
        self.layer_original_ranks = layer_original_ranks or {}
        self.default_original_rank = default_original_rank
        self.group_size = group_size
        self.residual_length = residual_length
        
        # Key quantizer (standard KIVI per-channel) - same for all layers
        self.key_quantizer = KIVIKeyQuantizer(n_bits=k_bits, group_size=group_size)
        
        # Per-layer Value quantizers (created lazily)
        self._value_quantizers: Dict[int, KIVIMixedPrecisionQuantizer] = {}
        
        # Cache storage
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._seen_tokens = 0
    
    def _get_value_quantizer(self, layer_idx: int) -> KIVIMixedPrecisionQuantizer:
        """Get or create a value quantizer for the given layer."""
        if layer_idx not in self._value_quantizers:
            # Get the original_rank for this layer
            original_rank = self.layer_original_ranks.get(layer_idx, self.default_original_rank)
            
            self._value_quantizers[layer_idx] = KIVIMixedPrecisionQuantizer(
                out_features=self.out_features,
                original_rank=original_rank,
                group_size=self.group_size,
            )
        
        return self._value_quantizers[layer_idx]
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self._cache:
            return 0
        key_quant, key_res, _, _ = self._cache[layer_idx]
        quant_len = key_quant.shape[1] if key_quant is not None else 0
        res_len = key_res.shape[1] if key_res is not None else 0
        return quant_len + res_len
    
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
        """Update cache with mixed-precision quantization for Value."""
        
        batch_size, seq_len, dim = key_states.shape if key_states.dim() == 3 else (
            key_states.shape[0], key_states.shape[2], key_states.shape[1] * key_states.shape[3]
        )
        
        # Flatten if 4D
        if key_states.dim() == 4:
            b, h, s, d = key_states.shape
            key_flat = key_states.transpose(1, 2).reshape(b, s, h * d)
            value_flat = value_states.transpose(1, 2).reshape(b, s, h * d)
            is_4d = True
        else:
            key_flat = key_states
            value_flat = value_states
            is_4d = False
        
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
                
                # Key: standard KIVI quantization
                key_quant = self.key_quantizer(key_to_quant)
                # Value: mixed-precision quantization (per-layer)
                value_quantizer = self._get_value_quantizer(layer_idx)
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
                    # Value: mixed-precision quantization (per-layer)
                    value_quantizer = self._get_value_quantizer(layer_idx)
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
        for layer_idx in self._cache:
            k_q, k_r, v_q, v_r = self._cache[layer_idx]
            if k_q is not None:
                k_q = k_q.index_select(0, beam_idx)
                v_q = v_q.index_select(0, beam_idx)
            if k_r is not None:
                k_r = k_r.index_select(0, beam_idx)
                v_r = v_r.index_select(0, beam_idx)
            self._cache[layer_idx] = (k_q, k_r, v_q, v_r)
    
    def get_cache_info(self) -> Dict[str, Any]:
        if not self._cache:
            return {"status": "empty"}
        
        layer_0 = self._cache.get(0)
        if layer_0 is None:
            return {"status": "no layer 0"}
        
        k_q, k_r, v_q, v_r = layer_0
        
        # Collect per-layer value quantizer info
        per_layer_info = {}
        for layer_idx, quantizer in self._value_quantizers.items():
            per_layer_info[layer_idx] = {
                "original_rank": quantizer.original_rank,
                "n_4bit": quantizer.n_4bit,
                "n_2bit": quantizer.n_2bit,
            }
        
        return {
            "num_layers": len(self._cache),
            "total_seq_len": self.get_seq_length(0),
            "quantized_len": k_q.shape[1] if k_q is not None else 0,
            "residual_len": k_r.shape[1] if k_r is not None else 0,
            "k_bits": self.k_bits,
            "per_layer_value_quant": per_layer_info,
            "layer_original_ranks": self.layer_original_ranks,
        }


class KIVIKeyQuantizer(nn.Module):
    """KIVI per-channel quantizer for Key."""
    
    def __init__(self, n_bits: int = 2, group_size: int = 128):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_bits >= 16:
            return x
        
        q_max = 2 ** self.n_bits - 1
        q_min = 0
        
        # Per-channel: transpose, quantize along seq dim, transpose back
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
                scale = (x_max - x_min).clamp(min=1e-5) / q_max
                zp = (-x_min / scale).round().clamp(q_min, q_max)
                
                x_q = (x_grouped / scale + zp).round().clamp(q_min, q_max)
                x_dq = (x_q - zp) * scale
                x_dq = x_dq.view(*leading, aligned)
                
                if x_rem.shape[-1] > 0:
                    x_dq = torch.cat([x_dq, x_rem], dim=-1)
                
                return x_dq.transpose(-1, -2)
        
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-5) / q_max
        zp = (-x_min / scale).round().clamp(q_min, q_max)
        x_q = (x / scale + zp).round().clamp(q_min, q_max)
        x_dq = (x_q - zp) * scale
        
        return x_dq.transpose(-1, -2)


# ============================================================================
# Factory Functions
# ============================================================================

def create_mixed_precision_cache(
    k_bits: int = 2,
    out_features: int = 4096,
    layer_original_ranks: Dict[int, int] = None,
    default_original_rank: int = 256,
    group_size: int = 128,
    residual_length: int = 128,
) -> KIVIMixedPrecisionCache:
    """
    Create a mixed-precision KIVI cache with per-layer rank support.
    
    Args:
        k_bits: Number of bits for Key quantization (per-channel)
        out_features: Output dimension D (typically num_kv_heads * head_dim)
        layer_original_ranks: Dict mapping layer_idx -> original_rank for that layer
            e.g., {0: 256, 1: 384, 2: 512, ...}
            This is used to calculate the 4bit/2bit split for each layer
        default_original_rank: Fallback rank if layer not in layer_original_ranks
        group_size: Group size for quantization
        residual_length: Number of recent tokens to keep in full precision
    
    Returns:
        KIVIMixedPrecisionCache configured with per-layer ranks
    """
    return KIVIMixedPrecisionCache(
        k_bits=k_bits,
        out_features=out_features,
        layer_original_ranks=layer_original_ranks,
        default_original_rank=default_original_rank,
        group_size=group_size,
        residual_length=residual_length,
    )


def calculate_mixed_precision_split(out_features: int, original_rank: int) -> Tuple[int, int]:
    """
    Calculate the 4bit/2bit split for mixed-precision quantization.
    
    Args:
        out_features: Full dimension (D)
        original_rank: Rank that would be used with 3bit quantization (r)
    
    Returns:
        (n_4bit, n_2bit): Number of features for each precision
    """
    D = out_features
    r = original_rank
    
    n_4bit = (3 * r - 2 * D) / 2
    
    if n_4bit < 0:
        return 0, D
    elif n_4bit > D:
        return D, 0
    else:
        n_4bit = int(n_4bit)
        return n_4bit, D - n_4bit
