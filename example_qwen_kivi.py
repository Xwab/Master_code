"""
Example: Using KIVI Quantized KV Cache with Qwen Models

This script demonstrates how to use KIVI-style KV cache quantization
with Qwen models for memory-efficient long-context inference.

Usage:
    python example_qwen_kivi.py --model Qwen/Qwen2-7B-Instruct --k_bits 2 --v_bits 2
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import KIVI cache
from modules.kivi_cache_general import KIVICache, create_kivi_cache
from modeling_qwen_kivi import (
    load_qwen_with_kivi,
    patch_qwen_model,
    QwenKIVIWrapper,
    get_kivi_memory_savings,
    print_kivi_stats,
)


def demo_basic_usage(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 1: Basic usage with load_qwen_with_kivi
    
    This is the simplest way to use KIVI with Qwen.
    """
    print("\n" + "=" * 70)
    print("Demo 1: Basic Usage with load_qwen_with_kivi")
    print("=" * 70)
    
    # Load model with KIVI support
    model, tokenizer = load_qwen_with_kivi(
        model_path,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=32,
        residual_length=128,
        torch_dtype=torch.float16,
    )
    
    # Generate - KIVI cache is automatically used!
    prompt = "Explain the concept of attention in transformer models in detail:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating with KIVI cache...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")
    
    return model, tokenizer


def demo_manual_cache(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 2: Manual cache control
    
    This shows how to manually create and manage the KIVI cache.
    """
    print("\n" + "=" * 70)
    print("Demo 2: Manual Cache Control")
    print("=" * 70)
    
    # Load standard model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Create KIVI cache manually
    cache = KIVICache(
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=32,
        residual_length=128,
    )
    
    prompt = "What are the benefits of quantizing the KV cache?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating with manual KIVI cache...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=cache,  # Pass KIVI cache explicitly
            use_cache=True,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")
    
    # Print cache statistics
    print_kivi_stats(cache)
    
    return model, tokenizer, cache


def demo_streaming_generation(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 3: Streaming generation with KIVI cache
    
    Shows token-by-token generation while reusing the cache.
    """
    print("\n" + "=" * 70)
    print("Demo 3: Streaming Generation")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Create KIVI cache
    cache = KIVICache(k_bits=k_bits, v_bits=v_bits, residual_length=128)
    
    prompt = "Write a short poem about AI:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    print(f"\nPrompt: {prompt}")
    print("\nStreaming output: ", end="", flush=True)
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(100):  # Generate up to 100 tokens
            outputs = model(
                input_ids=generated_ids if cache.get_seq_length() == 0 else generated_ids[:, -1:],
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


def demo_long_context(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 4: Long context with memory savings analysis
    
    Shows KIVI benefits for long sequences.
    """
    print("\n" + "=" * 70)
    print("Demo 4: Long Context Memory Analysis")
    print("=" * 70)
    
    # Calculate memory savings for different sequence lengths
    print("\nMemory Savings Analysis:")
    print("-" * 60)
    
    for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        stats = get_kivi_memory_savings(
            seq_length=seq_len,
            k_bits=k_bits,
            v_bits=v_bits,
            residual_length=128,
        )
        print(f"Seq={seq_len:>6}: {stats['compression_ratio']:>6} compression, "
              f"avg bits={stats['total_bits_avg']}, "
              f"memory={stats['memory_ratio']}")
    
    print("-" * 60)
    
    # Actual generation test if model is available
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Create a long prompt
        long_text = "The quick brown fox jumps over the lazy dog. " * 100
        long_prompt = f"Summarize this text:\n{long_text}\n\nSummary:"
        
        inputs = tokenizer(long_prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs.input_ids.to(model.device)
        
        print(f"\nLong prompt length: {input_ids.shape[1]} tokens")
        
        # Generate with KIVI cache
        cache = KIVICache(k_bits=k_bits, v_bits=v_bits, residual_length=128)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                past_key_values=cache,
                use_cache=True,
                max_new_tokens=50,
                do_sample=False,
            )
        
        print(f"Generated {outputs.shape[1] - input_ids.shape[1]} new tokens")
        print_kivi_stats(cache)
        
    except Exception as e:
        print(f"\nSkipping actual generation: {e}")


def demo_wrapper_class(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 5: Using QwenKIVIWrapper
    
    Alternative approach using a wrapper class.
    """
    print("\n" + "=" * 70)
    print("Demo 5: QwenKIVIWrapper")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Wrap model
    model = QwenKIVIWrapper(
        base_model,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=32,
        residual_length=128,
    )
    
    prompt = "What is machine learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
    
    print(f"\nPrompt: {prompt}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")


def demo_comparison(model_path: str, k_bits: int = 2, v_bits: int = 2):
    """
    Demo 6: Compare KIVI vs standard cache
    
    Shows that outputs are similar but memory is saved.
    """
    print("\n" + "=" * 70)
    print("Demo 6: KIVI vs Standard Cache Comparison")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with standard cache
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
    print("\n2. KIVI Cache (2-bit):")
    cache = KIVICache(k_bits=k_bits, v_bits=v_bits, residual_length=128)
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


def main():
    parser = argparse.ArgumentParser(description="KIVI Cache Demo for Qwen")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                        help="Model name or path")
    parser.add_argument("--k_bits", type=int, default=2, choices=[2, 4, 8, 16],
                        help="Key quantization bits")
    parser.add_argument("--v_bits", type=int, default=2, choices=[2, 4, 8, 16],
                        help="Value quantization bits")
    parser.add_argument("--demo", type=str, default="all",
                        choices=["basic", "manual", "streaming", "long", "wrapper", "comparison", "all"],
                        help="Which demo to run")
    args = parser.parse_args()
    
    print("=" * 70)
    print("KIVI Quantized KV Cache Demo for Qwen")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Key bits: {args.k_bits}, Value bits: {args.v_bits}")
    
    demos = {
        "basic": demo_basic_usage,
        "manual": demo_manual_cache,
        "streaming": demo_streaming_generation,
        "long": demo_long_context,
        "wrapper": demo_wrapper_class,
        "comparison": demo_comparison,
    }
    
    if args.demo == "all":
        # Run memory analysis (no model needed)
        demo_long_context(args.model, args.k_bits, args.v_bits)
    else:
        demos[args.demo](args.model, args.k_bits, args.v_bits)
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
