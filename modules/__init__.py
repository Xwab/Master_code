"""
Modules for ALRD (Adaptive Low-Rank Decomposition) with KIVI quantization.
"""

from .quant_utils import (
    Quantizer,
    Quantizer2,
    quantize_tensor,
    quantize_tensor_forward,
    # KIVI-style quantization
    KIVIKeyQuantizer,
    KIVIValueQuantizer,
    KIVIMixedQuantizer,
    kivi_quantize_per_channel,
    kivi_quantize_per_token,
)

from .kivi_cache import (
    KIVIQuantizer,
    KIVICache,
    KIVIDynamicCache,
    KIVILatentCache,
    create_kivi_quantizers,
    create_kivi_cache,
)

from .kivi_mixed_cache import (
    KIVIMixedPrecisionQuantizer,
    ALRDLinear_KIVI_Value_FullRank_Mixed,
    KIVIMixedPrecisionCache,
    create_mixed_precision_cache,
    calculate_mixed_precision_split,
)

from .kivi_mixed_cache_v2 import (
    KIVIKeyQuantizerV2,
    KIVIMixedValueQuantizerV2,
    ALRDLinear_KIVI_Value_FullRank_MixedV2,
    KIVIMixedPrecisionCacheV2,
    create_mixed_precision_cache_v2,
    calculate_mixed_precision_split_v2,
)

from .svd_linear import SVDLinear
from .hadamard_utils import apply_hadamard

# INT8 matmul for value reconstruction
from .int8_matmul import (
    Int8Quantizer,
    Int8Linear,
    Int8ValueReconstructor,
    create_int8_value_reconstructor,
    convert_alinear_to_int8,
    INT8_SUPPORTED,
)

__all__ = [
    # Legacy quantization
    "Quantizer",
    "Quantizer2",
    "quantize_tensor",
    "quantize_tensor_forward",
    # KIVI quantization
    "KIVIKeyQuantizer",
    "KIVIValueQuantizer", 
    "KIVIMixedQuantizer",
    "kivi_quantize_per_channel",
    "kivi_quantize_per_token",
    "KIVIQuantizer",
    "KIVICache",
    "KIVIDynamicCache",
    "KIVILatentCache",
    "create_kivi_quantizers",
    "create_kivi_cache",
    # Mixed-precision KIVI (full-rank) - 4bit/2bit
    "KIVIMixedPrecisionQuantizer",
    "ALRDLinear_KIVI_Value_FullRank_Mixed",
    "KIVIMixedPrecisionCache",
    "create_mixed_precision_cache",
    "calculate_mixed_precision_split",
    # Mixed-precision KIVI V2 (full-rank) - Key:5bit, Value:6bit/4bit
    "KIVIKeyQuantizerV2",
    "KIVIMixedValueQuantizerV2",
    "ALRDLinear_KIVI_Value_FullRank_MixedV2",
    "KIVIMixedPrecisionCacheV2",
    "create_mixed_precision_cache_v2",
    "calculate_mixed_precision_split_v2",
    # SVD
    "SVDLinear",
    "apply_hadamard",
    # INT8 matmul
    "Int8Quantizer",
    "Int8Linear",
    "Int8ValueReconstructor",
    "create_int8_value_reconstructor",
    "convert_alinear_to_int8",
    "INT8_SUPPORTED",
]
