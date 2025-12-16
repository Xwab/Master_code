# KIVI Integration for Low-Rank KV Cache

本文档说明如何将 KIVI 的量化方法集成到低秩分解的 KV 缓存压缩方案中。

## 架构概述

```
原始架构 (你的代码):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ hidden_states│ --> │   BLinear   │ --> │  低秩特征   │ --> │   ALinear   │ --> │ K/V states │
└─────────────┘     │ (降维到rank) │     │ (存入Cache) │     │ (恢复维度)  │     └─────────────┘
                    └─────────────┘     └─────────────┘     └─────────────┘

新架构 (KIVI 集成):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ hidden_states│ --> │   BLinear   │ --> │ KIVI量化    │ --> │  量化Cache  │ --> │   ALinear   │ --> │ K/V states │
└─────────────┘     └─────────────┘     │ (2bit)      │     └─────────────┘     └─────────────┘
                                        └─────────────┘
                                              │
                                              ├── Key: per-channel 量化
                                              └── Value: per-token 量化
```

## 文件结构

```
workspace/
├── configuration_alrd_llama.py    # 配置类 (新增)
├── modeling_alrd_llama.py         # 模型架构 (已修改)
├── modules/
│   ├── __init__.py                # 模块导出 (已修改)
│   ├── quant_utils.py             # 量化工具 (已修改，添加KIVI函数)
│   ├── kivi_cache.py              # KIVI缓存类 (新增)
│   ├── svd_linear.py              # SVD分解层
│   └── hadamard_utils.py          # 哈达玛变换工具
└── KIVI_INTEGRATION.md            # 本文档
```

## 使用方法

### 1. 基本使用

```python
from configuration_alrd_llama import ALRDLlamaConfig
from modeling_alrd_llama import ALRDLlamaForCausalLM

# 创建配置
config = ALRDLlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

# 设置低秩分解的 rank (每层可以不同)
config.truncation_ranks = {
    "model.layers.0.self_attn.k_proj": 256,
    "model.layers.0.self_attn.v_proj": 256,
    # ... 其他层
}

# 启用 KIVI 量化
config.use_kivi = True
config.k_bits = 2          # Key 使用 2bit
config.v_bits = 2          # Value 使用 2bit
config.group_size = 128    # 量化分组大小
config.residual_length = 32  # 保留全精度的最近 token 数

# 加载模型
model = ALRDLlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

### 2. 量化参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k_bits` | 2 | Key 缓存的量化位数 (2/3/4/8) |
| `v_bits` | 2 | Value 缓存的量化位数 (2/3/4/8) |
| `group_size` | 128 | 量化分组大小，0 表示不分组 |
| `residual_length` | 32 | 保留全精度的最近 token 数量 |
| `use_kivi` | True | 是否使用 KIVI 量化 |

### 3. KIVI 量化原理

**Key Cache (per-channel 量化)**:
- 沿 token 维度计算 min/max
- 每个 channel (特征维度) 共享量化参数
- 适合 attention score 计算，保持不同 token 间的相对关系

```python
# Key: [batch, seq_len, rank] -> 按 channel 量化
# 实际操作: transpose -> quantize along seq_len -> transpose back
key_latent = k_proj.BLinear(hidden_states)  # [B, L, rank]
key_quant = kivi_quantize_per_channel(key_latent, n_bits=2, group_size=128)
```

**Value Cache (per-token 量化)**:
- 沿 hidden 维度计算 min/max
- 每个 token 独立量化
- 适合加权求和操作

```python
# Value: [batch, seq_len, rank] -> 按 token 量化
value_latent = v_proj.BLinear(hidden_states)  # [B, L, rank]
value_quant = kivi_quantize_per_token(value_latent, n_bits=2, group_size=128)
```

### 4. 高级配置：混合精度量化

KIVI 的一个关键设计是保留最近的 token 为全精度：

```python
from modules.quant_utils import KIVIMixedQuantizer

# 创建混合精度量化器
key_quantizer = KIVIMixedQuantizer(
    n_bits=2,
    group_size=128,
    residual_length=32,   # 最近 32 个 token 保持 FP16
    per_channel=True,     # Key 使用 per-channel
)

value_quantizer = KIVIMixedQuantizer(
    n_bits=2,
    group_size=128,
    residual_length=32,
    per_channel=False,    # Value 使用 per-token
)
```

### 5. 自定义 KIVI Cache

如果需要更细粒度的控制，可以使用 `KIVIDynamicCache`:

```python
from modules.kivi_cache import KIVIDynamicCache, create_kivi_quantizers

# 创建量化器
key_quantizer, value_quantizer = create_kivi_quantizers(
    k_bits=2,
    v_bits=2,
    group_size=128,
    residual_length=32,
)

# 创建动态缓存
cache = KIVIDynamicCache(
    num_layers=32,
    k_bits=2,
    v_bits=2,
    group_size=128,
    residual_length=32,
)

# 在推理中使用
for layer_idx in range(num_layers):
    key_latent = k_proj.BLinear(hidden_states)
    value_latent = v_proj.BLinear(hidden_states)
    
    # 更新缓存（自动处理量化和残差）
    all_keys, all_values = cache.update(
        key_latent, value_latent, layer_idx,
        key_quantizer, value_quantizer
    )
```

## 压缩率计算

假设:
- 原始 head_dim = 128
- 低秩 rank = 64 (50% 压缩)
- KIVI 量化 2bit

压缩率计算:
```
原始: 128 * 16bit = 2048 bits
低秩: 64 * 16bit = 1024 bits (2x 压缩)
低秩 + KIVI 2bit: 64 * 2bit = 128 bits (16x 压缩)

总压缩率: 2048 / 128 = 16x
```

实际中考虑量化参数开销:
```
量化参数 (scale + zero_point): 约 0.5 bits/element (分组量化)
有效 bits/element: 2 + 0.5 = 2.5 bits
实际压缩率: 2048 / (64 * 2.5) = ~13x
```

## 与原始 KIVI 的区别

| 方面 | 原始 KIVI | 你的实现 |
|------|----------|----------|
| 量化对象 | 完整 K/V states | 低秩 latent (BLinear 输出) |
| 维度 | head_dim (128) | rank (可配置，如 64) |
| 额外压缩 | 无 | 低秩分解 ~2x |
| 量化方式 | 相同 | 相同 (per-channel/per-token) |

## 注意事项

1. **Group Size 选择**: 
   - `group_size` 应该能整除 `rank`
   - 较小的 group_size 提供更好的精度但增加参数开销
   - 推荐: 64 或 128

2. **Residual Length**:
   - 设置为 32-128 之间
   - 太小会影响长文本质量
   - 太大会减少压缩效果

3. **兼容性**:
   - 设置 `use_kivi=False` 可回退到原始简单量化
   - 所有 `quantize_latent` 和 `quantize_latent_mixed` 接口保持兼容

4. **性能**:
   - 当前实现使用 PyTorch 原生操作
   - 如需进一步优化，可以添加 CUDA kernel (参考原始 KIVI 的 `quant/` 目录)

## 调试建议

```python
# 检查量化效果
from modules.quant_utils import kivi_quantize_per_channel, kivi_quantize_per_token

# 测试 Key 量化
key_latent = torch.randn(1, 100, 64)  # [batch, seq_len, rank]
key_quant, scale, zp = kivi_quantize_per_channel(key_latent, n_bits=2, group_size=64)
error = (key_latent - key_quant).abs().mean()
print(f"Key quantization error: {error:.6f}")

# 测试 Value 量化
value_latent = torch.randn(1, 100, 64)
value_quant, scale, zp = kivi_quantize_per_token(value_latent, n_bits=2, group_size=64)
error = (value_latent - value_quant).abs().mean()
print(f"Value quantization error: {error:.6f}")
```
