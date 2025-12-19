# KIVI Quantized KV Cache for Qwen Models

本文档说明如何在 Qwen 系列模型上使用 KIVI 风格的 KV Cache 量化。

## 概述

KIVI (Key-Value Inference) 是一种针对 KV Cache 的量化方法：
- **Key**: Per-channel 量化（沿 token 维度）
- **Value**: Per-token 量化（沿 head_dim 维度）
- **Residual**: 最近的 token 保持全精度

由于 KIVI 官方仓库不支持 Qwen，我们实现了一个兼容版本。

## 快速开始

### 方法 1: 使用 `load_qwen_with_kivi`（推荐）

```python
from modeling_qwen_kivi import load_qwen_with_kivi

# 加载模型
model, tokenizer = load_qwen_with_kivi(
    "Qwen/Qwen2-7B-Instruct",
    k_bits=2,      # Key 使用 2bit 量化
    v_bits=2,      # Value 使用 2bit 量化
    group_size=32,
    residual_length=128,
)

# 生成 - KIVI cache 自动启用！
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### 方法 2: 手动创建 KIVI Cache

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.kivi_cache_general import KIVICache

# 加载标准模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# 创建 KIVI cache
cache = KIVICache(
    k_bits=2,
    v_bits=2,
    group_size=32,
    residual_length=128,
)

# 生成
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    past_key_values=cache,  # 传入 KIVI cache
    use_cache=True,
    max_new_tokens=100,
)
```

### 方法 3: Patch 现有模型

```python
from modeling_qwen_kivi import patch_qwen_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", ...)

# Patch 模型
patch_qwen_model(
    model,
    k_bits=2,
    v_bits=2,
    auto_inject_cache=True,  # 自动在 generate() 中使用 KIVI
)

# 现在 generate() 会自动使用 KIVI cache
outputs = model.generate(**inputs, max_new_tokens=100)
```

## 支持的模型

- Qwen2 / Qwen2.5 (所有尺寸)
- Qwen (原版)
- Llama / Llama2 / Llama3
- 任何使用标准 transformers Cache 接口的模型

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k_bits` | 2 | Key 量化位数 (2, 4, 8, 16) |
| `v_bits` | 2 | Value 量化位数 (2, 4, 8, 16) |
| `group_size` | 32 | 量化分组大小（越小精度越高，开销越大） |
| `residual_length` | 128 | 保持全精度的最近 token 数量 |

## 量化配置推荐

| 场景 | k_bits | v_bits | group_size | residual |
|------|--------|--------|------------|----------|
| 极致压缩 | 2 | 2 | 32 | 128 |
| 平衡 | 4 | 4 | 32 | 128 |
| 高质量 | 4 | 4 | 16 | 256 |
| 仅节省内存 | 8 | 8 | 32 | 128 |

## 内存节省分析

```python
from modeling_qwen_kivi import get_kivi_memory_savings

# 计算不同序列长度的内存节省
for seq_len in [1024, 4096, 8192, 32768]:
    stats = get_kivi_memory_savings(seq_len, k_bits=2, v_bits=2)
    print(f"Seq={seq_len}: {stats['compression_ratio']} compression")
```

输出示例：
```
Seq=1024: 4.57x compression
Seq=4096: 6.40x compression
Seq=8192: 7.11x compression
Seq=32768: 7.76x compression
```

## PPL 评估

```bash
python eval_ppl_qwen_kivi.py \
    --model Qwen/Qwen2-7B-Instruct \
    --k_bits 2 \
    --v_bits 2 \
    --max_tokens 4096
```

## 查看 Cache 状态

```python
from modules.kivi_cache_general import KIVICache

cache = KIVICache(k_bits=2, v_bits=2)

# 生成后查看状态
outputs = model.generate(..., past_key_values=cache)

# 打印统计信息
info = cache.get_cache_info()
print(info)
# {
#     'num_layers': 32,
#     'total_seq_len': 512,
#     'quantized_len': 384,
#     'residual_len': 128,
#     'k_bits': 2,
#     'v_bits': 2,
#     'effective_bits': '3.50',
#     'memory_ratio': '21.88%',
# }
```

## 与官方 KIVI 的区别

| 特性 | 官方 KIVI | 本实现 |
|------|-----------|--------|
| Qwen 支持 | ❌ | ✅ |
| Llama 支持 | ✅ | ✅ |
| Fake 量化 | ✅ | ✅ |
| Triton 加速 | ✅ | ❌ (计划中) |
| 内存节省 | ✅ (真实) | ❌ (Fake) |
| 精度等效 | - | ✅ |

注意：当前实现是 **Fake 量化**，即量化后立即反量化回 FP16。这用于评估量化对精度的影响，但不会真正节省内存。真实内存节省需要实现 Triton/CUDA 内核。

## 流式生成示例

```python
cache = KIVICache(k_bits=2, v_bits=2, residual_length=128)

input_ids = tokenizer("Tell me a story:", return_tensors="pt").input_ids.to(model.device)
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

print()
print(f"\nCache info: {cache.get_cache_info()}")
```

## 常见问题

### Q: 为什么使用 KIVI cache 后 PPL 没有变化或变好了？

A: 这可能是因为：
1. 测试序列太短，大部分 token 在 residual 中（全精度）
2. 评估方法不正确（每个 chunk 都创建新 cache）
3. 量化误差恰好帮助了某些情况

建议使用 `--incremental` 模式评估，这更接近真实推理场景。

### Q: 如何验证 KIVI cache 正在被使用？

```python
# 生成后检查 cache 类型和状态
print(type(outputs.past_key_values))  # 应该是 KIVICache
print(outputs.past_key_values.get_cache_info())  # 查看量化统计
```

### Q: 支持 beam search 吗？

是的，`KIVICache` 实现了 `reorder_cache` 方法，支持 beam search。

## 文件结构

```
/workspace/
├── modules/
│   ├── kivi_cache_general.py  # 通用 KIVI cache (本实现)
│   ├── kivi_cache.py          # 低秩 ALRD 专用 KIVI cache
│   └── ...
├── modeling_qwen_kivi.py      # Qwen KIVI 适配层
├── example_qwen_kivi.py       # 使用示例
├── eval_ppl_qwen_kivi.py      # PPL 评估脚本
└── KIVI_QWEN_USAGE.md         # 本文档
```
