"""
Example: Using KIVI Quantized KV Cache with Llama Models

This script demonstrates how to use KIVI-style KV cache quantization
with Llama models (no low-rank decomposition, just quantized cache).

Usage:
    python example_llama_kivi.py --model meta-llama/Llama-3.1-8B-Instruct --k_bits 2 --v_bits 2
"""

import argparse
import torch
from transformers import AutoTokenizer

# Import KIVI Llama model
from modeling_llama_kivi import (
    LlamaForCausalLM_KIVI,
    LlamaKIVIConfig,
    load_llama_kivi,
    print_kivi_stats,
)
from modules.kivi_cache_general import KIVICache


def demo_basic_usage(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 1: Basic usage with load_llama_kivi helper
    """
    print("\n" + "=" * 70)
    print("Demo 1: Basic Usage with load_llama_kivi")
    print("=" * 70)
    
    # Load model with KIVI support
    model, tokenizer = load_llama_kivi(
        model_path,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=32,
        residual_length=128,
        torch_dtype=torch.float16,
    )
    
    prompt = "Explain the theory of relativity in simple terms:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\nPrompt: {prompt}")
    
    # Create KIVI cache manually
    cache = model.create_kivi_cache()
    
    print("\nGenerating with KIVI cache...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")
    
    # Print cache stats
    print_kivi_stats(cache)
    
    return model, tokenizer


def demo_from_pretrained(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 2: Using from_pretrained directly
    """
    print("\n" + "=" * 70)
    print("Demo 2: Using from_pretrained directly")
    print("=" * 70)
    
    # Load with KIVI parameters
    model = LlamaForCausalLM_KIVI.from_pretrained(
        model_path,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=32,
        residual_length=128,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    prompt = "What are the benefits of renewable energy?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\nPrompt: {prompt}")
    
    # Create cache manually
    cache = model.create_kivi_cache()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            max_new_tokens=80,
        )
    
    print(f"\nResponse: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    print_kivi_stats(cache)
    
    return model, tokenizer


def demo_auto_inject(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 3: Auto-inject KIVI cache
    """
    print("\n" + "=" * 70)
    print("Demo 3: Auto-inject KIVI Cache")
    print("=" * 70)
    
    model, tokenizer = load_llama_kivi(model_path, k_bits=k_bits, v_bits=v_bits)
    
    # Enable auto-injection
    model.use_kivi_cache = True
    
    prompt = "Write a haiku about programming:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\nPrompt: {prompt}")
    print("(KIVI cache will be auto-injected)")
    
    with torch.no_grad():
        # No need to pass cache - it will be auto-created
        outputs = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=50,
        )
    
    print(f"\nResponse: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    
    return model, tokenizer


def demo_streaming(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 4: Streaming generation with KIVI cache
    """
    print("\n" + "=" * 70)
    print("Demo 4: Streaming Generation")
    print("=" * 70)
    
    model, tokenizer = load_llama_kivi(model_path, k_bits=k_bits, v_bits=v_bits)
    
    prompt = "Once upon a time, in a land far away,"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    print(f"\nPrompt: {prompt}")
    print("\nStreaming output: ", end="", flush=True)
    
    # Create KIVI cache
    cache = model.create_kivi_cache()
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(80):
            # First call: full input, subsequent: only last token
            if cache.get_seq_length() == 0:
                curr_input = generated_ids
            else:
                curr_input = generated_ids[:, -1:]
            
            outputs = model(
                input_ids=curr_input,
                past_key_values=cache,
                use_cache=True,
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Print token
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_text, end="", flush=True)
            
            # Stop on EOS
            if next_token[0].item() == tokenizer.eos_token_id:
                break
    
    print("\n")
    print_kivi_stats(cache)
    
    return model, tokenizer


def demo_comparison(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 5: Compare KIVI vs standard cache
    """
    print("\n" + "=" * 70)
    print("Demo 5: KIVI vs Standard Cache Comparison")
    print("=" * 70)
    
    model, tokenizer = load_llama_kivi(model_path, k_bits=k_bits, v_bits=v_bits)
    
    prompt = "The most important thing about machine learning is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with standard cache (DynamicCache)
    print("\n1. Standard DynamicCache:")
    torch.manual_seed(42)
    with torch.no_grad():
        outputs_std = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
        )
    print(tokenizer.decode(outputs_std[0], skip_special_tokens=True))
    
    # Generate with KIVI cache
    print(f"\n2. KIVI Cache (k={k_bits}bit, v={v_bits}bit):")
    cache = model.create_kivi_cache()
    torch.manual_seed(42)
    with torch.no_grad():
        outputs_kivi = model.generate(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
        )
    print(tokenizer.decode(outputs_kivi[0], skip_special_tokens=True))
    
    print_kivi_stats(cache)
    
    return model, tokenizer


def demo_memory_analysis():
    """
    Demo 6: Memory savings analysis (no model needed)
    """
    print("\n" + "=" * 70)
    print("Demo 6: Memory Savings Analysis")
    print("=" * 70)
    
    def calculate_memory_savings(seq_len, k_bits, v_bits, residual_length):
        """Calculate effective bits per element."""
        if seq_len <= residual_length:
            quant_len = 0
            residual_len = seq_len
        else:
            quant_len = seq_len - residual_length
            residual_len = residual_length
        
        avg_k_bits = (quant_len * k_bits + residual_len * 16) / seq_len
        avg_v_bits = (quant_len * v_bits + residual_len * 16) / seq_len
        avg_bits = (avg_k_bits + avg_v_bits) / 2
        
        return {
            "seq_len": seq_len,
            "quant_len": quant_len,
            "residual_len": residual_len,
            "avg_bits": avg_bits,
            "compression": 16 / avg_bits,
            "memory_ratio": avg_bits / 16,
        }
    
    print("\nMemory Savings for KIVI-2bit (k=2, v=2, residual=128):")
    print("-" * 70)
    print(f"{'Seq Length':<12} {'Quant Len':<12} {'Avg Bits':<12} {'Compression':<12} {'Memory':<12}")
    print("-" * 70)
    
    for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        stats = calculate_memory_savings(seq_len, k_bits=2, v_bits=2, residual_length=128)
        print(f"{stats['seq_len']:<12} {stats['quant_len']:<12} {stats['avg_bits']:<12.2f} "
              f"{stats['compression']:<12.2f}x {stats['memory_ratio']:<12.1%}")
    
    print("-" * 70)
    print("\n对比不同量化配置 (Seq Length = 4096):")
    print("-" * 55)
    
    configs = [
        (2, 2, "KIVI-2bit"),
        (4, 4, "KIVI-4bit"),
        (4, 2, "K4V2"),
        (2, 4, "K2V4"),
        (8, 8, "KIVI-8bit"),
    ]
    
    seq_len = 4096
    print(f"{'Config':<15} {'Avg Bits':<12} {'Compression':<12} {'Memory':<12}")
    print("-" * 55)
    
    for k_bits, v_bits, name in configs:
        stats = calculate_memory_savings(seq_len, k_bits, v_bits, residual_length=128)
        print(f"{name:<15} {stats['avg_bits']:<12.2f} {stats['compression']:<12.2f}x {stats['memory_ratio']:<12.1%}")


def main():
    parser = argparse.ArgumentParser(description="KIVI Cache Demo for Llama")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name or path")
    parser.add_argument("--k_bits", type=int, default=2, choices=[2, 4, 8, 16],
                        help="Key quantization bits")
    parser.add_argument("--v_bits", type=int, default=2, choices=[2, 4, 8, 16],
                        help="Value quantization bits")
    parser.add_argument("--demo", type=str, default="memory",
                        choices=["basic", "pretrained", "auto", "streaming", "comparison", "memory", "all"],
                        help="Which demo to run")
    args = parser.parse_args()
    
    print("=" * 70)
    print("KIVI Quantized KV Cache Demo for Llama (No Low-Rank)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Key bits: {args.k_bits}, Value bits: {args.v_bits}")
    
    demos = {
        "basic": demo_basic_usage,
        "pretrained": demo_from_pretrained,
        "auto": demo_auto_inject,
        "streaming": demo_streaming,
        "comparison": demo_comparison,
    }
    
    if args.demo == "memory":
        demo_memory_analysis()
    elif args.demo == "all":
        demo_memory_analysis()
        for name, func in demos.items():
            try:
                func(args.model, args.k_bits, args.v_bits)
            except Exception as e:
                print(f"\nDemo {name} failed: {e}")
    else:
        demos[args.demo](args.model, args.k_bits, args.v_bits)
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
