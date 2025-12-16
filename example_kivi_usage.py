"""
Example: Using KIVI quantization with low-rank KV cache.

This example shows how to:
1. Configure ALRD model with KIVI quantization
2. Create and use KIVILatentCache for inference
3. Compare with standard cache
"""

import torch
from typing import Dict

# Assuming these are properly set up
# from configuration_alrd_llama import ALRDLlamaConfig
# from modeling_alrd_llama import ALRDLlamaForCausalLM
from modules.kivi_cache import KIVILatentCache, create_kivi_cache, KIVIQuantizer
from modules.quant_utils import kivi_quantize_per_channel, kivi_quantize_per_token


def example_basic_usage():
    """Basic usage example."""
    
    # === Step 1: Configure the model ===
    # config = ALRDLlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Set truncation ranks for each layer
    # truncation_ranks = {}
    # for i in range(32):  # 32 layers for Llama-2-7b
    #     truncation_ranks[f"model.layers.{i}.self_attn.k_proj"] = 256  # rank=256
    #     truncation_ranks[f"model.layers.{i}.self_attn.v_proj"] = 256
    # config.truncation_ranks = truncation_ranks
    
    # Enable KIVI quantization
    # config.use_kivi = True
    # config.k_bits = 2          # Key: 2bit quantization
    # config.v_bits = 2          # Value: 2bit quantization
    # config.group_size = 128    # Quantization group size
    # config.residual_length = 32  # Keep last 32 tokens in FP16
    
    # === Step 2: Load model ===
    # model = ALRDLlamaForCausalLM.from_pretrained(
    #     "meta-llama/Llama-2-7b-hf",
    #     config=config,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    
    # === Step 3: Create KIVI cache ===
    # kivi_cache = model.create_kivi_cache()
    
    # === Step 4: Generate with KIVI cache ===
    # input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
    # outputs = model.generate(
    #     input_ids,
    #     past_key_values=kivi_cache,
    #     use_cache=True,
    #     max_new_tokens=100,
    # )
    
    print("Basic usage example (requires model weights)")


def example_kivi_cache_standalone():
    """Demonstrate KIVILatentCache without the full model."""
    
    print("=" * 60)
    print("KIVILatentCache Standalone Example")
    print("=" * 60)
    
    # Create cache
    cache = create_kivi_cache(
        k_bits=2,
        v_bits=2,
        group_size=64,
        residual_length=16,
    )
    
    # Simulate prefill: add 100 tokens
    batch_size = 1
    rank = 128  # low-rank dimension
    seq_len = 100
    
    # Simulate key/value latents from BLinear
    key_latent = torch.randn(batch_size, seq_len, rank)
    value_latent = torch.randn(batch_size, seq_len, rank)
    
    # Update cache for layer 0
    layer_idx = 0
    all_keys, all_values = cache.update(key_latent, value_latent, layer_idx)
    
    print(f"\nAfter prefill ({seq_len} tokens):")
    print(f"  Cache seq length: {cache.get_seq_length(layer_idx)}")
    print(f"  All keys shape: {all_keys.shape}")
    print(f"  All values shape: {all_values.shape}")
    
    # Simulate decode: add tokens one by one
    for step in range(5):
        new_key = torch.randn(batch_size, 1, rank)
        new_value = torch.randn(batch_size, 1, rank)
        
        all_keys, all_values = cache.update(new_key, new_value, layer_idx)
        print(f"  After decode step {step+1}: seq_len = {cache.get_seq_length(layer_idx)}")
    
    print("\nKIVI cache successfully manages quantized + residual tokens!")


def example_quantization_comparison():
    """Compare per-channel vs per-token quantization."""
    
    print("=" * 60)
    print("Quantization Method Comparison")
    print("=" * 60)
    
    # Create test tensor
    x = torch.randn(1, 64, 128)  # [batch, seq_len, rank]
    
    # Per-channel quantization (for Key)
    x_channel, _, _ = kivi_quantize_per_channel(x, n_bits=2, group_size=64)
    error_channel = (x - x_channel).abs().mean().item()
    
    # Per-token quantization (for Value)
    x_token, _, _ = kivi_quantize_per_token(x, n_bits=2, group_size=64)
    error_token = (x - x_token).abs().mean().item()
    
    print(f"\nInput shape: {x.shape}")
    print(f"Quantization bits: 2")
    print(f"Group size: 64")
    print(f"\nPer-channel (Key) quantization error: {error_channel:.6f}")
    print(f"Per-token (Value) quantization error: {error_token:.6f}")
    
    # Compare different bit widths
    print("\n--- Error vs Bit Width ---")
    for bits in [2, 3, 4, 8]:
        x_ch, _, _ = kivi_quantize_per_channel(x, n_bits=bits, group_size=64)
        x_tk, _, _ = kivi_quantize_per_token(x, n_bits=bits, group_size=64)
        print(f"  {bits}-bit: per-channel={abs(x - x_ch).mean():.6f}, "
              f"per-token={abs(x - x_tk).mean():.6f}")


def example_memory_estimation():
    """Estimate memory savings with KIVI + low-rank."""
    
    print("=" * 60)
    print("Memory Estimation")
    print("=" * 60)
    
    # Model parameters
    num_layers = 32
    num_kv_heads = 8
    head_dim = 128
    seq_len = 4096
    batch_size = 1
    
    # Low-rank parameters
    rank = 64  # 50% compression
    
    # Calculate memory
    original_bytes = 2 * num_layers * num_kv_heads * seq_len * head_dim * 2  # FP16
    lowrank_bytes = 2 * num_layers * num_kv_heads * seq_len * rank * 2  # FP16
    
    # KIVI parameters
    k_bits = 2
    v_bits = 2
    residual_length = 32
    
    # KIVI memory (simplified, ignoring scale/zp overhead)
    quant_seq_len = seq_len - residual_length
    kivi_quant_bytes = 2 * num_layers * num_kv_heads * quant_seq_len * rank * (k_bits + v_bits) / 8 / 2
    kivi_residual_bytes = 2 * num_layers * num_kv_heads * residual_length * rank * 2
    kivi_total_bytes = kivi_quant_bytes + kivi_residual_bytes
    
    print(f"\nConfiguration:")
    print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")
    print(f"  Sequence length: {seq_len}, Batch size: {batch_size}")
    print(f"  Low-rank: {rank}, KIVI bits: {k_bits}/{v_bits}")
    
    print(f"\nMemory Usage:")
    print(f"  Original (FP16):     {original_bytes / 1024 / 1024:.2f} MB")
    print(f"  Low-rank only:       {lowrank_bytes / 1024 / 1024:.2f} MB ({original_bytes/lowrank_bytes:.1f}x compression)")
    print(f"  Low-rank + KIVI:     {kivi_total_bytes / 1024 / 1024:.2f} MB ({original_bytes/kivi_total_bytes:.1f}x compression)")


def example_quantizer_classes():
    """Demonstrate KIVIQuantizer class usage."""
    
    print("=" * 60)
    print("KIVIQuantizer Class Usage")
    print("=" * 60)
    
    # Create quantizers
    key_quantizer = KIVIQuantizer(n_bits=2, group_size=64, per_channel=True)
    value_quantizer = KIVIQuantizer(n_bits=2, group_size=64, per_channel=False)
    
    # Test data
    x = torch.randn(2, 100, 128)
    
    # Quantize
    key_quant = key_quantizer(x)
    value_quant = value_quantizer(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Key quantized shape: {key_quant.shape}")
    print(f"Value quantized shape: {value_quant.shape}")
    
    # Get scale and zero_point
    key_dequant, key_scale, key_zp = key_quantizer.quantize(x)
    value_dequant, value_scale, value_zp = value_quantizer.quantize(x)
    
    print(f"\nKey scale shape: {key_scale.shape if key_scale is not None else 'None'}")
    print(f"Value scale shape: {value_scale.shape if value_scale is not None else 'None'}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KIVI Integration Examples")
    print("=" * 60 + "\n")
    
    example_basic_usage()
    print()
    
    example_kivi_cache_standalone()
    print()
    
    example_quantization_comparison()
    print()
    
    example_memory_estimation()
    print()
    
    example_quantizer_classes()
