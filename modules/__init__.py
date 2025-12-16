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
    create_kivi_quantizers,
)

from .svd_linear import SVDLinear
from .hadamard_utils import apply_hadamard

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
    "create_kivi_quantizers",
    # SVD
    "SVDLinear",
    "apply_hadamard",
]
