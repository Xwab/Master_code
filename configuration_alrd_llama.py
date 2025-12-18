"""
Configuration for ALRD (Adaptive Low-Rank Decomposition) LLaMA with KIVI quantization.
"""

from transformers import LlamaConfig


class ALRDLlamaConfig(LlamaConfig):
    """
    Configuration class for ALRD LLaMA model with KIVI-style KV cache quantization.
    
    Args:
        truncation_ranks: Dict mapping layer names to their truncation ranks
            e.g., {"model.layers.0.self_attn.k_proj": 256, ...}
        
        # KIVI quantization parameters
        k_bits: Number of bits for Key cache quantization (default: 2)
        v_bits: Number of bits for Value cache quantization (default: 2)
        group_size: Group size for quantization (default: 128)
        residual_length: Number of recent tokens to keep in full precision (default: 32)
        use_kivi: Whether to use KIVI-style quantization (default: True)
        
        # Mixed-precision Value quantization (truncated rank)
        use_mixed_precision_value: Whether to use mixed 4bit/2bit for Value (default: False)
        value_target_ratios: Dict mapping layer names to target compression ratios
            e.g., {"model.layers.0.self_attn.v_proj": 0.2, ...}
            If not specified, uses default_value_target_ratio
        default_value_target_ratio: Default target compression ratio for Value (default: 0.25)
        
        # Full-rank mixed-precision Value quantization (NEW)
        use_fullrank_mixed_value: Whether to use full-rank + 4bit/2bit mixed quantization
            Instead of: low-rank (r) + 3bit uniform quantization
            Use: full-rank (D) + 4bit/2bit mixed quantization with same compression
            
            Compression equivalence:
            - Original: 3r bits (rank r, 3bit per element)
            - New: 4*n_4bit + 2*n_2bit bits (full D dimensions)
            
            Split calculation:
            - n_4bit = (3r - 2D) / 2  (features with larger singular values)
            - n_2bit = (4D - 3r) / 2  (features with smaller singular values)
            - Valid when: 2D/3 <= r <= 4D/3
        
        # ALinear weight quantization (KIVI-style)
        a_weight_bits: Number of bits for ALinear weight quantization (default: 8)
        a_weight_group_size: Group size for ALinear weight quantization (default: 128)
        
        # Legacy quantization parameters (for backward compatibility)
        latent_quant_bits: Bits for latent quantization when not using KIVI
    """
    
    model_type = "alrd_llama"
    
    def __init__(
        self,
        truncation_ranks: dict = None,
        # KIVI quantization parameters
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 128,
        residual_length: int = 32,
        use_kivi: bool = True,
        # Mixed-precision Value quantization (truncated rank)
        use_mixed_precision_value: bool = False,
        value_target_ratios: dict = None,
        default_value_target_ratio: float = 0.25,
        # Full-rank mixed-precision Value quantization (NEW)
        use_fullrank_mixed_value: bool = False,
        # ALinear weight quantization
        a_weight_bits: int = 8,
        a_weight_group_size: int = 128,
        # Legacy parameters
        latent_quant_bits: int = 3,
        latent_quant_sym: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.truncation_ranks = truncation_ranks or {}
        
        # KIVI parameters
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.use_kivi = use_kivi
        
        # Mixed-precision Value parameters (truncated rank)
        self.use_mixed_precision_value = use_mixed_precision_value
        self.value_target_ratios = value_target_ratios or {}
        self.default_value_target_ratio = default_value_target_ratio
        
        # Full-rank mixed-precision Value (NEW)
        self.use_fullrank_mixed_value = use_fullrank_mixed_value
        
        # Mixed precision quantization options (V1: Key 2bit, Value 4bit/2bit)
        # match_compression=True: 匹配低秩+3bit的压缩率 (只有当 rank >= 2D/3 时有效)
        # match_compression=False: 使用固定比例的高精度
        self.mixed_match_compression = kwargs.pop('mixed_match_compression', True)
        self.mixed_high_precision_ratio = kwargs.pop('mixed_high_precision_ratio', 0.25)
        self.mixed_high_bits = kwargs.pop('mixed_high_bits', 4)
        self.mixed_low_bits = kwargs.pop('mixed_low_bits', 2)
        
        # Mixed precision V2 options (Key 5bit, Value 6bit/4bit)
        self.use_mixed_v2 = kwargs.pop('use_mixed_v2', False)  # 使用 V2 版本
        self.mixed_v2_k_bits = kwargs.pop('mixed_v2_k_bits', 5)  # Key 用 5bit
        self.mixed_v2_original_avg_bits = kwargs.pop('mixed_v2_original_avg_bits', 5.0)  # 原始方案平均bits
        self.mixed_v2_high_bits = kwargs.pop('mixed_v2_high_bits', 6)  # Value 高精度用 6bit
        self.mixed_v2_low_bits = kwargs.pop('mixed_v2_low_bits', 4)  # Value 低精度用 4bit
        
        # ALinear weight quantization
        self.a_weight_bits = a_weight_bits
        self.a_weight_group_size = a_weight_group_size
        
        # Legacy parameters
        self.latent_quant_bits = latent_quant_bits
        self.latent_quant_sym = latent_quant_sym
