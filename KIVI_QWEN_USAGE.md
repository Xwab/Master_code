# KIVI Quantized KV Cache for Qwen Models (无低秩分解)

本文档说明如何在 Qwen 系列模型上使用 KIVI 风格的 KV Cache 量化。

**注意：这个实现不使用低秩分解，保留原始完整模型参数，只对 KV Cache 进行 KIVI 量化。**

## 概述

KIVI (Key-Value Inference) 是一种针对 KV Cache 的量化方法：
- **Key**: Per-channel 量化（沿 token 维度）
- **Value**: Per-token 量化（沿 head_dim 维度）
- **Residual**: 最近的 token 保持全精度

由于 KIVI 官方仓库不支持 Qwen，我们实现了一个兼容版本。

## 文件结构

```
/workspace/
├── modeling_qwen_kivi.py          # Qwen KIVI 模型实现 (主文件)
├── modules/
│   └── kivi_cache_general.py      # 通用 KIVI Cache 类
├── example_qwen_kivi.py           # 使用示例
├── eval_ppl_qwen_kivi.py          # PPL 评估脚本
└── KIVI_QWEN_USAGE.md             # 本文档
```

## 快速开始

### 方法 1: 使用 `load_qwen_kivi` 辅助函数（推荐）

```python
from modeling_qwen_kivi import load_qwen_kivi

# 加载模型
model, tokenizer = load_qwen_kivi(
    "Qwen/Qwen2-7B-Instruct",
    k_bits=2,      # Key 使用 2bit 量化
    v_bits=2,      # Value 使用 2bit 量化
    group_size=32,
    residual_length=128,
    torch_dtype=torch.float16,
)

# 创建 KIVI cache
cache = model.create_kivi_cache()

# 生成
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    past_key_values=cache,
    use_cache=True,
    max_new_tokens=100,
)
print(tokenizer.decode(outputs[0]))

# 查看 cache 状态
from modeling_qwen_kivi import print_kivi_stats
print_kivi_stats(cache)
```

### 方法 2: 使用 `from_pretrained` 直接加载

```python
from modeling_qwen_kivi import Qwen2ForCausalLM_KIVI, Qwen2KIVIConfig
from transformers import AutoTokenizer

# 加载配置
config = Qwen2KIVIConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
config.k_bits = 2
config.v_bits = 2
config.group_size = 32
config.residual_length = 128

# 加载模型
model = Qwen2ForCausalLM_KIVI.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# 创建 KIVI cache 并生成
cache = model.create_kivi_cache()
outputs = model.generate(..., past_key_values=cache, use_cache=True)
```

### 方法 3: 自动注入 KIVI Cache

```python
from modeling_qwen_kivi import load_qwen_kivi

model, tokenizer = load_qwen_kivi("Qwen/Qwen2-7B-Instruct", k_bits=2, v_bits=2)

# 启用自动注入
model.use_kivi_cache = True

# 现在 generate() 会自动使用 KIVI cache
outputs = model.generate(**inputs, use_cache=True, max_new_tokens=100)
```

## 支持的模型

- Qwen2 / Qwen2.5 (所有尺寸)
- Qwen (原版)

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k_bits` | 2 | Key 量化位数 (2, 4, 8, 16) |
| `v_bits` | 2 | Value 量化位数 (2, 4, 8, 16) |
| `group_size` | 32 | 量化分组大小（越小精度越高，开销越大） |
| `residual_length` | 128 | 保持全精度的最近 token 数量 |

## 量化配置推荐

| 场景 | k_bits | v_bits | group_size | residual | 说明 |
|------|--------|--------|------------|----------|------|
| 极致压缩 | 2 | 2 | 32 | 128 | 最大压缩，适合长序列 |
| 平衡 | 4 | 4 | 32 | 128 | 较好的精度/压缩平衡 |
| 高质量 | 4 | 4 | 16 | 256 | 更小 group_size 提高精度 |
| 保守 | 8 | 8 | 32 | 128 | 精度损失最小 |

## 内存节省分析

```python
# 运行内存分析 demo
python example_qwen_kivi.py --demo memory
```

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

## PPL 评估

```bash
python eval_ppl_qwen_kivi.py \
    --model Qwen/Qwen2-7B-Instruct \
    --k_bits 2 \
    --v_bits 2 \
    --max_tokens 4096
```

## 流式生成示例

```python
from modeling_qwen_kivi import load_qwen_kivi

model, tokenizer = load_qwen_kivi("Qwen/Qwen2-7B-Instruct", k_bits=2, v_bits=2)

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
from modeling_qwen_kivi import print_kivi_stats
print_kivi_stats(cache)
```

## 核心类说明

### `Qwen2KIVIConfig`

继承自 `Qwen2Config`，添加 KIVI 参数：

```python
class Qwen2KIVIConfig(Qwen2Config):
    k_bits: int = 2
    v_bits: int = 2
    group_size: int = 32
    residual_length: int = 128
    use_kivi: bool = True
```

### `Qwen2ForCausalLM_KIVI`

继承自 `Qwen2ForCausalLM`，主要变化：
- 使用 `Qwen2KIVIModel` 替代 `Qwen2Model`
- Attention 层使用 `Qwen2KIVISdpaAttention` 或 `Qwen2KIVIFlashAttention2`
- 添加 `create_kivi_cache()` 方法
- 支持自动注入 KIVI cache (`use_kivi_cache = True`)

### `KIVICache`

兼容 transformers `Cache` 接口的 KIVI cache：

```python
class KIVICache(Cache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        更新 cache，自动处理量化：
        - 旧 token: 量化存储
        - 最近 residual_length 个 token: 全精度存储
        """
        ...
    
    def get_cache_info(self) -> Dict:
        """获取 cache 统计信息"""
        ...
```

## 与官方 KIVI 的区别

| 特性 | 官方 KIVI | 本实现 |
|------|-----------|--------|
| Qwen 支持 | ❌ | ✅ |
| Llama 支持 | ✅ | ✅ |
| 低秩分解 | ✅ | ❌ (使用完整参数) |
| Fake 量化 | ✅ | ✅ |
| Triton 加速 | ✅ | ❌ |
| 实际内存节省 | ✅ | ❌ (Fake 量化) |
| 精度评估 | ✅ | ✅ |

**重要说明**：当前实现是 **Fake 量化**，即量化后立即反量化回 FP16。这用于评估量化对精度的影响，但**不会真正节省内存**。真实内存节省需要实现 Triton/CUDA 内核存储 INT 数据。

## 常见问题

### Q: 如何验证 KIVI cache 正在被使用？

```python
# 生成后检查 cache 类型和状态
cache = model.create_kivi_cache()
outputs = model.generate(..., past_key_values=cache)

print(type(cache))  # KIVICache
print(cache.get_cache_info())  # 查看量化统计
```

### Q: 支持 beam search 吗？

是的，`KIVICache` 实现了 `reorder_cache` 方法，支持 beam search。

### Q: 为什么 PPL 变化不大？

1. 测试序列太短，大部分 token 在 residual 中（全精度）
2. KIVI 量化对较短序列影响较小
3. 建议在长序列 (4K+) 上测试

### Q: 如何与 modeling_alrd_llama.py 的低秩+KIVI 结合？

`modeling_qwen_kivi.py` 不使用低秩分解。如果需要低秩+KIVI 的 Qwen 版本，需要类似 `modeling_alrd_llama.py` 的实现，用 ALRDLinear 替换 k_proj 和 v_proj。

## 运行示例

```bash
# 内存分析（不需要 GPU）
python example_qwen_kivi.py --demo memory

# 基本使用
python example_qwen_kivi.py --model Qwen/Qwen2-1.5B-Instruct --demo basic

# 对比 KIVI vs 标准 cache
python example_qwen_kivi.py --model Qwen/Qwen2-1.5B-Instruct --demo comparison

# 流式生成
python example_qwen_kivi.py --model Qwen/Qwen2-1.5B-Instruct --demo streaming
```
