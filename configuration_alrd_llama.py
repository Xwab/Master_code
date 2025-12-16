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
        
        # Mixed-precision Value quantization
        use_mixed_precision_value: Whether to use mixed 4bit/2bit for Value (default: False)
        value_target_ratios: Dict mapping layer names to target compression ratios
            e.g., {"model.layers.0.self_attn.v_proj": 0.2, ...}
            If not specified, uses default_value_target_ratio
        default_value_target_ratio: Default target compression ratio for Value (default: 0.25)
        
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
        # Mixed-precision Value quantization
        use_mixed_precision_value: bool = False,
        value_target_ratios: dict = None,
        default_value_target_ratio: float = 0.25,
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
        
        # Mixed-precision Value parameters
        self.use_mixed_precision_value = use_mixed_precision_value
        self.value_target_ratios = value_target_ratios or {}
        self.default_value_target_ratio = default_value_target_ratio
        
        # Legacy parameters
        self.latent_quant_bits = latent_quant_bits
        self.latent_quant_sym = latent_quant_sym
