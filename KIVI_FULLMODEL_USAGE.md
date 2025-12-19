# KIVI Quantized KV Cache for Full Models (无低秩分解)

本文档说明如何在 Llama 和 Qwen 系列模型上使用 KIVI 风格的 KV Cache 量化。

**注意：这些实现不使用低秩分解，保留原始完整模型参数，只对 KV Cache 进行 KIVI 量化。**

## 文件结构

```
/workspace/
├── modeling_llama_kivi.py         # Llama KIVI 模型实现
├── modeling_qwen_kivi.py          # Qwen KIVI 模型实现
├── modules/
│   └── kivi_cache_general.py      # 通用 KIVI Cache 类
├── example_llama_kivi.py          # Llama 使用示例
├── example_qwen_kivi.py           # Qwen 使用示例
└── KIVI_FULLMODEL_USAGE.md        # 本文档
```

## 支持的模型

| 模型系列 | 文件 | 测试版本 |
|----------|------|----------|
| Llama | `modeling_llama_kivi.py` | Llama-2, Llama-3, Llama-3.1, Llama-3.2 |
| Qwen | `modeling_qwen_kivi.py` | Qwen2, Qwen2.5 |

## 快速开始

### Llama 模型

```python
from modeling_llama_kivi import load_llama_kivi, LlamaForCausalLM_KIVI

# 方法 1: 使用辅助函数
model, tokenizer = load_llama_kivi(
    "meta-llama/Llama-3.1-8B-Instruct",
    k_bits=2,
    v_bits=2,
)

# 创建 KIVI cache
cache = model.create_kivi_cache()

# 生成
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, past_key_values=cache, use_cache=True, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))

# 方法 2: 使用 from_pretrained
model = LlamaForCausalLM_KIVI.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    k_bits=2,
    v_bits=2,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

### Qwen 模型

```python
from modeling_qwen_kivi import load_qwen_kivi, Qwen2ForCausalLM_KIVI

# 方法 1: 使用辅助函数
model, tokenizer = load_qwen_kivi(
    "Qwen/Qwen2-7B-Instruct",
    k_bits=2,
    v_bits=2,
)

# 创建 KIVI cache
cache = model.create_kivi_cache()

# 生成
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, past_key_values=cache, use_cache=True, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))

# 方法 2: 使用 from_pretrained
model = Qwen2ForCausalLM_KIVI.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    k_bits=2,
    v_bits=2,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

## 自动注入 KIVI Cache

```python
# 启用自动注入
model.use_kivi_cache = True

# 现在 generate() 会自动使用 KIVI cache
outputs = model.generate(**inputs, use_cache=True, max_new_tokens=100)
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k_bits` | 2 | Key 量化位数 (2, 4, 8, 16) |
| `v_bits` | 2 | Value 量化位数 (2, 4, 8, 16) |
| `group_size` | 32 | 量化分组大小 |
| `residual_length` | 128 | 保持全精度的最近 token 数量 |
| `use_kivi` | True | 是否启用 KIVI 量化 |

## 核心类对比

| 类名 | Llama 版本 | Qwen 版本 |
|------|------------|-----------|
| Config | `LlamaKIVIConfig` | `Qwen2KIVIConfig` |
| Model | `LlamaForCausalLM_KIVI` | `Qwen2ForCausalLM_KIVI` |
| Attention (SDPA) | `LlamaKIVISdpaAttention` | `Qwen2KIVISdpaAttention` |
| Attention (Flash) | `LlamaKIVIFlashAttention2` | `Qwen2KIVIFlashAttention2` |
| Decoder Layer | `LlamaKIVIDecoderLayer` | `Qwen2KIVIDecoderLayer` |
| Base Model | `LlamaKIVIModel` | `Qwen2KIVIModel` |
| Helper | `load_llama_kivi()` | `load_qwen_kivi()` |

## 内存压缩率

KIVI-2bit (k=2, v=2, residual=128) 的压缩率：

| Seq Length | Avg Bits | Compression | Memory |
|------------|----------|-------------|--------|
| 256 | 9.00 | 1.78x | 56.2% |
| 512 | 5.50 | 2.91x | 34.4% |
| 1024 | 3.75 | 4.27x | 23.4% |
| 2048 | 2.88 | 5.57x | 18.0% |
| 4096 | 2.44 | 6.56x | 15.3% |
| 8192 | 2.22 | 7.21x | 13.9% |
| 32768 | 2.06 | 7.78x | 12.9% |

## 流式生成示例

```python
# 创建 KIVI cache
cache = model.create_kivi_cache()

prompt = "Tell me a story:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
generated_ids = input_ids.clone()

for _ in range(100):
    # 首次调用传入完整序列，后续只传入最后一个 token
    if cache.get_seq_length() == 0:
        curr_input = generated_ids
    else:
        curr_input = generated_ids[:, -1:]
    
    outputs = model(
        input_ids=curr_input,
        past_key_values=cache,
        use_cache=True,
    )
    
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    
    # 打印 token
    print(tokenizer.decode(next_token[0]), end="", flush=True)
    
    if next_token[0].item() == tokenizer.eos_token_id:
        break

# 查看 cache 状态
from modeling_llama_kivi import print_kivi_stats  # 或 modeling_qwen_kivi
print_kivi_stats(cache)
```

## 运行示例

```bash
# Llama 内存分析
python example_llama_kivi.py --demo memory

# Llama 基本使用
python example_llama_kivi.py --model meta-llama/Llama-3.2-1B-Instruct --demo basic

# Qwen 内存分析
python example_qwen_kivi.py --demo memory

# Qwen 基本使用
python example_qwen_kivi.py --model Qwen/Qwen2-1.5B-Instruct --demo basic
```

## 与 KIVI 官方的区别

| 特性 | 官方 KIVI | 本实现 (Llama/Qwen) |
|------|-----------|---------------------|
| Qwen 支持 | ❌ | ✅ |
| Llama 支持 | ✅ | ✅ |
| 低秩分解 | ✅ | ❌ (使用完整参数) |
| Fake 量化 | ✅ | ✅ |
| Triton 加速 | ✅ | ❌ |
| 实际内存节省 | ✅ | ❌ (Fake 量化) |

**重要说明**：当前实现是 **Fake 量化**，用于评估量化对精度的影响，但**不会真正节省内存**。

## 与低秩+KIVI 版本的区别

| 特性 | modeling_alrd_llama.py | modeling_llama_kivi.py |
|------|------------------------|------------------------|
| 低秩分解 | ✅ (ALRDLinear) | ❌ |
| K/V 投影 | BLinear + ALinear | 原始 k_proj, v_proj |
| KIVI 量化 | ✅ (对 latent) | ✅ (对原始 KV) |
| 用途 | 低秩压缩 + 量化 | 仅量化 |

## 常见问题

### Q: 如何验证 KIVI cache 正在被使用？

```python
cache = model.create_kivi_cache()
outputs = model.generate(..., past_key_values=cache)

print(type(cache))  # KIVICache
print(cache.get_cache_info())  # 查看量化统计
```

### Q: 支持 beam search 吗？

是的，`KIVICache` 实现了 `reorder_cache` 方法。

### Q: 如何在 lm_eval 等框架中使用？

```python
model.use_kivi_cache = True  # 启用自动注入
# 然后正常使用 lm_eval
```

### Q: 为什么 PPL 变化不大？

- 测试序列太短，大部分 token 在 residual 中
- 建议在长序列 (4K+) 上测试
