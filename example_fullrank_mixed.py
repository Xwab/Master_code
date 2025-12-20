"""
Example: Full-Rank Mixed-Precision KIVI Quantization for Value Cache

This example demonstrates the new quantization scheme:
- Original: low-rank (r) + 3bit uniform quantization
- New: full-rank (D) + 4bit/2bit mixed quantization

The compression ratio is kept the same:
  3r = 4 * n_4bit + 2 * n_2bit

Where:
  n_4bit = (3r - 2D) / 2  (features with larger singular values → 4bit)
  n_2bit = (4D - 3r) / 2  (features with smaller singular values → 2bit)

Valid range: 2D/3 <= r <= 4D/3
"""

import torch
from transformers import AutoTokenizer

# Import our custom classes
from configuration_alrd_llama import ALRDLlamaConfig
from modeling_alrd_llama import ALRDLlamaForCausalLM
from modules.kivi_mixed_cache import (
    calculate_mixed_precision_split,
    KIVIMixedPrecisionQuantizer,
    ALRDLinear_KIVI_Value_FullRank_Mixed,
)


def calculate_compression_ratio_analysis():
    """
    Analyze the 4bit/2bit split for different rank values.
    
    For a dimension D=1024, show how different ranks affect the split.
    """
    print("=" * 70)
    print("Compression Ratio Analysis")
    print("=" * 70)
    
    D = 1024  # Full dimension (out_features)
    
    # Valid range: 2D/3 <= r <= 4D/3
    # i.e., 683 <= r <= 1365
    
    print(f"\nDimension D = {D}")
    print(f"Valid rank range: {2*D//3} to {4*D//3}")
    print()
    
    ranks = [256, 384, 512, 640, 768, 896, 1024]
    
    print(f"{'Rank':>8} | {'n_4bit':>8} | {'n_2bit':>8} | {'Orig Bits':>10} | {'New Bits':>10} | {'Status':>12}")
    print("-" * 70)
    
    for r in ranks:
        n_4bit, n_2bit = calculate_mixed_precision_split(D, r)
        
        orig_bits = 3 * r  # Original: r elements × 3bit
        new_bits = 4 * n_4bit + 2 * n_2bit  # New: mixed
        
        orig_avg = orig_bits / D  # Bits per element (original)
        new_avg = new_bits / D  # Bits per element (new)
        
        if n_4bit < 0:
            status = "all 2bit"
        elif n_4bit > D:
            status = "all 4bit"
        else:
            status = "mixed"
        
        print(f"{r:>8} | {n_4bit:>8} | {n_2bit:>8} | {orig_avg:>10.2f} | {new_avg:>10.2f} | {status:>12}")
    
    print()


def test_mixed_precision_quantizer():
    """
    Test the KIVIMixedPrecisionQuantizer directly.
    """
    print("=" * 70)
    print("Testing KIVIMixedPrecisionQuantizer")
    print("=" * 70)
    
    out_features = 1024
    original_rank = 512  # Would use 3bit with this rank
    group_size = 128
    
    quantizer = KIVIMixedPrecisionQuantizer(
        out_features=out_features,
        original_rank=original_rank,
        group_size=group_size,
    )
    
    # Create test input
    batch_size = 2
    seq_len = 256
    x = torch.randn(batch_size, seq_len, out_features)
    
    # Apply mixed-precision quantization
    x_quant = quantizer(x)
    
    # Check error
    error = (x - x_quant).abs().mean()
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {x_quant.shape}")
    print(f"Mean absolute error: {error.item():.6f}")
    
    # Error for 4bit and 2bit parts
    if quantizer.n_4bit > 0 and quantizer.n_4bit < out_features:
        x_4bit = x[..., :quantizer.n_4bit]
        x_4bit_quant = x_quant[..., :quantizer.n_4bit]
        error_4bit = (x_4bit - x_4bit_quant).abs().mean()
        
        x_2bit = x[..., quantizer.n_4bit:]
        x_2bit_quant = x_quant[..., quantizer.n_4bit:]
        error_2bit = (x_2bit - x_2bit_quant).abs().mean()
        
        print(f"4bit part error: {error_4bit.item():.6f} (first {quantizer.n_4bit} features)")
        print(f"2bit part error: {error_2bit.item():.6f} (last {quantizer.n_2bit} features)")
    
    print()


def test_fullrank_linear():
    """
    Test the ALRDLinear_KIVI_Value_FullRank_Mixed layer.
    """
    print("=" * 70)
    print("Testing ALRDLinear_KIVI_Value_FullRank_Mixed")
    print("=" * 70)
    
    in_features = 4096
    out_features = 1024
    original_rank = 512  # Would use 3bit with this rank
    
    layer = ALRDLinear_KIVI_Value_FullRank_Mixed(
        in_features=in_features,
        out_features=out_features,
        original_rank=original_rank,
        bias=True,
        group_size=128,
        residual_length=128,
    )
    
    # Create test input
    batch_size = 2
    seq_len = 256
    x = torch.randn(batch_size, seq_len, in_features)
    
    # Forward pass
    y = layer(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"BLinear weight shape: {layer.BLinear.weight.shape}")
    print(f"ALinear weight shape: {layer.ALinear.weight.shape}")
    print(f"n_4bit: {layer.n_4bit}, n_2bit: {layer.n_2bit}")
    
    # Test Hadamard fusion
    print("\nApplying Hadamard transform...")
    layer.fuse_hadamard()
    
    y_after_hada = layer(x)
    print(f"Output after Hadamard: {y_after_hada.shape}")
    
    print()


def demo_model_usage():
    """
    Demonstrate how to use the full-rank mixed-precision model.
    """
    print("=" * 70)
    print("Model Usage Demo")
    print("=" * 70)
    
    print("""
# 1. Configure the model with full-rank mixed-precision Value

from configuration_alrd_llama import ALRDLlamaConfig
from modeling_alrd_llama import ALRDLlamaForCausalLM

config = ALRDLlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# Set truncation ranks (the ranks that WOULD be used with 3bit)
config.truncation_ranks = {
    "model.layers.0.self_attn.k_proj": 256,
    "model.layers.0.self_attn.v_proj": 256,
    # ... for all layers
}

# Enable KIVI quantization
config.use_kivi = True
config.k_bits = 2  # Key uses 2bit per-channel quantization

# Enable full-rank mixed-precision for Value (NEW!)
config.use_fullrank_mixed_value = True
# This will:
# - Keep all D features (no truncation)
# - Use 4bit for features with larger singular values
# - Use 2bit for features with smaller singular values
# - Match compression: 3r = 4*n_4bit + 2*n_2bit

# Load the model
model = ALRDLlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config,
    torch_dtype=torch.float16,
)

# 2. Create KIVI cache for inference
kivi_cache = model.create_kivi_cache()  # For standard KIVI
# OR
mixed_cache = model.create_mixed_precision_cache()  # For mixed precision

# 3. Generate with the cache
outputs = model.generate(
    input_ids,
    past_key_values=kivi_cache,  # or mixed_cache
    use_cache=True,
    max_new_tokens=100,
)
    """)


def compare_compression_schemes():
    """
    Compare the original vs new compression schemes.
    """
    print("=" * 70)
    print("Compression Scheme Comparison")
    print("=" * 70)
    
    D = 1024  # Full dimension
    r = 512   # Truncation rank
    L = 4096  # Sequence length
    residual = 128  # Full precision residual tokens
    
    print(f"\nParameters: D={D}, r={r}, L={L}, residual={residual}")
    print()
    
    # Original: low-rank + 3bit
    # Storage: r elements per token, 3bit each + quantization overhead
    L_quant = L - residual
    
    # Original scheme
    orig_bits_per_token = r * 3  # 3bit per element, r elements
    orig_total_bits = L_quant * orig_bits_per_token + residual * r * 16  # quantized + residual FP16
    orig_avg_bits = orig_total_bits / (L * r)
    
    print("Original Scheme: Low-Rank (r={}) + 3bit Uniform".format(r))
    print(f"  Bits per token (quantized): {orig_bits_per_token}")
    print(f"  Total bits: {orig_total_bits:,}")
    print(f"  Average bits/element: {orig_avg_bits:.2f}")
    print()
    
    # New scheme: full-rank + 4bit/2bit mixed
    n_4bit, n_2bit = calculate_mixed_precision_split(D, r)
    if n_4bit < 0:
        n_4bit = 0
        n_2bit = D
    
    new_bits_per_token = 4 * n_4bit + 2 * n_2bit
    new_total_bits = L_quant * new_bits_per_token + residual * D * 16  # quantized + residual FP16
    new_avg_bits = new_total_bits / (L * D)
    
    print("New Scheme: Full-Rank (D={}) + 4bit/2bit Mixed".format(D))
    print(f"  n_4bit: {n_4bit}, n_2bit: {n_2bit}")
    print(f"  Bits per token (quantized): {new_bits_per_token}")
    print(f"  Total bits: {new_total_bits:,}")
    print(f"  Average bits/element: {new_avg_bits:.2f}")
    print()
    
    # Compression ratio comparison
    orig_compression = orig_total_bits / (L * D * 16)  # vs full FP16
    new_compression = new_total_bits / (L * D * 16)  # vs full FP16
    
    print("Compression Ratios (vs FP16):")
    print(f"  Original: {orig_compression:.4f} ({orig_compression*100:.2f}%)")
    print(f"  New: {new_compression:.4f} ({new_compression*100:.2f}%)")
    print()
    
    # Key insight
    print("Key Insight:")
    print("  Original stores fewer elements (r < D) but same bits/element")
    print("  New stores more elements (D > r) but uses mixed precision")
    print("  The 4bit/2bit split is designed so total storage is equivalent")
    print()


if __name__ == "__main__":
    calculate_compression_ratio_analysis()
    test_mixed_precision_quantizer()
    test_fullrank_linear()
    compare_compression_schemes()
    demo_model_usage()
