# INT8 量化操作使用指南

## 概述

`int8_ops.py` 提供了完整的 INT8 量化矩阵乘法实现：

1. **INT8Quantizer** - 返回真正的 INT8 值（不是 fake quant）
2. **INT8Linear** - 预量化权重的 INT8 线性层
3. **Triton Kernel** - 高效的 CUDA kernel（如果 Triton 可用）

## 与原 Quantizer 的区别

```python
# 原 Quantizer (fake quant) - 返回 float
from modules.quant_utils import Quantizer
quantizer = Quantizer(n_bits=8, ...)
x_fake_quant = quantizer(x)  # 返回 float，值已量化再反量化

# 新 INT8Quantizer (真正量化) - 返回 int8 + scale
from int8_ops import INT8Quantizer
x_int8, scale = INT8Quantizer.quantize_per_token(x)  # 返回 int8 和 scale
x_restored = INT8Quantizer.dequantize(x_int8, scale)  # 需要时再反量化
```

## 核心组件

### 1. INT8Quantizer

```python
from int8_ops import INT8Quantizer

# Per-token 量化 (用于 Value latent)
# 每个 token 独立计算 scale
x = torch.randn(2, 512, 256)  # (batch, seq, hidden)
x_int8, scale = INT8Quantizer.quantize_per_token(x)
# x_int8: (2, 512, 256) int8
# scale:  (2, 512, 1) float

# Per-channel 量化 (用于重建矩阵)
# 每个输出通道独立计算 scale
w = torch.randn(1024, 256)  # (out_features, in_features)
w_int8, scale = INT8Quantizer.quantize_per_channel(w)
# w_int8: (1024, 256) int8
# scale:  (1024, 1) float

# 反量化
x_restored = INT8Quantizer.dequantize(x_int8, scale)
```

### 2. INT8Linear

```python
from int8_ops import INT8Linear

# 创建 INT8 线性层（权重会被预量化）
weight = torch.randn(1024, 256)  # (out_features, in_features)
bias = torch.randn(1024)
layer = INT8Linear(weight, bias, backend="auto")

# 前向传播
x = torch.randn(2, 512, 256, dtype=torch.float16, device='cuda')
output = layer(x)  # (2, 512, 1024)

# 内部流程:
# 1. x_int8, x_scale = quantize_per_token(x)  # 在线量化输入
# 2. out_int32 = x_int8 @ w_int8.T            # INT8 矩阵乘法
# 3. out = out_int32 * x_scale * w_scale.T    # 反量化
# 4. out = out + bias
```

### 3. INT8LinearFunction

```python
from int8_ops import INT8Quantizer, INT8LinearFunction

# 预量化权重
w = torch.randn(1024, 256, device='cuda')
w_int8, w_scale = INT8Quantizer.quantize_per_channel(w)

# 使用函数接口
x = torch.randn(2, 512, 256, dtype=torch.float16, device='cuda')
bias = torch.randn(1024, device='cuda')

output = INT8LinearFunction.apply(
    x,        # 输入
    w_int8,   # 预量化的权重
    w_scale,  # 权重 scale
    bias,     # 偏置
    "auto"    # 后端: "triton", "torch", "auto"
)
```

## 后端选择

| 后端 | 条件 | 性能 |
|------|------|------|
| `triton` | Triton 已安装 + CUDA | 最快 |
| `torch` | PyTorch 2.0+ + CUDA | 较快 |
| `fallback` | 任意环境 | 无加速 |

```python
# 自动选择最佳后端
layer = INT8Linear(weight, backend="auto")

# 强制使用 Triton
layer = INT8Linear(weight, backend="triton")

# 强制使用 torch._int_mm
layer = INT8Linear(weight, backend="torch")
```

## 在低秩压缩模型中使用

### ALRDLinearINT8 的工作流程

```python
# modeling_alrd_llama_int8.py

class ALRDLinearINT8(nn.Module):
    def __init__(self, in_features, out_features, rank, ...):
        self.BLinear = nn.Linear(in_features, rank)  # 低秩投影
        self.ALinear = nn.Linear(rank, out_features) # 重建矩阵
        
        # INT8 权重 (延迟初始化)
        self.A_int8 = None
        self.A_scale = None
    
    def prepare_int8_weights(self):
        # 预量化重建矩阵 A (per-channel)
        w = self.ALinear.weight.data
        self.A_int8, self.A_scale = INT8Quantizer.quantize_per_channel(w)
    
    def forward_int8(self, latent):
        # latent: (batch, seq, rank) - 来自 BLinear 的输出
        
        # 使用 INT8 矩阵乘法进行重建
        # 内部: 量化 latent (per-token) -> INT8 matmul -> 反量化
        output = INT8LinearFunction.apply(
            latent,
            self.A_int8,
            self.A_scale,
            self.ALinear.bias,
            self.backend
        )
        return output
```

### Attention 中的使用

```python
# 在 CustomLlamaFlashAttention2INT8 中

def forward(self, hidden_states, ...):
    # Query: 正常 FP16
    query_states = self.q_proj(hidden_states)
    
    # Key: 低秩投影 + FP16 重建
    key_latent = self.k_proj.BLinear(hidden_states)
    key_latent = self.k_proj.quantize_latent(key_latent)  # fake quant for storage
    key_states = self.k_proj.ALinear(key_latent)  # FP16 重建
    
    # Value: 低秩投影 + INT8 重建 !!!
    value_latent = self.v_proj.BLinear(hidden_states)
    value_latent = self.v_proj.quantize_latent(value_latent)  # fake quant for storage
    
    # 更新 KV Cache (存储低秩表示)
    if past_key_value is not None:
        key_latent, value_latent = past_key_value.update(key_latent, value_latent, ...)
    
    # 重建
    key_states = self.k_proj.ALinear(key_latent)      # FP16
    value_states = self.v_proj.forward_int8(value_latent)  # INT8 !!!
    
    # ... rest of attention
```

## Triton Kernel 详解

### 量化 Kernel

```python
@triton.jit
def _int8_quantize_per_token_kernel(
    x_ptr,        # 输入 float 指针
    x_int8_ptr,   # 输出 int8 指针
    scale_ptr,    # 输出 scale 指针
    n_cols,       # 列数
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)  # 每个 block 处理一行
    
    # 加载一行数据
    col_offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, ...)
    
    # 计算 scale = max(|x|) / 127
    amax = tl.max(tl.abs(x))
    scale = amax / 127.0
    
    # 量化: x_int8 = round(x / scale)
    x_int8 = tl.libdevice.rint(x / scale)
    x_int8 = tl.minimum(tl.maximum(x_int8, -128.0), 127.0)
    
    # 存储
    tl.store(x_int8_ptr + row_idx * n_cols + col_offsets, x_int8.to(tl.int8))
    tl.store(scale_ptr + row_idx, scale)
```

### 矩阵乘法 Kernel

```python
@triton.jit
def _int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    ...
):
    # 分块矩阵乘法
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(...)  # int8
        b = tl.load(...)  # int8
        acc += tl.dot(a, b)  # int32 累加
    
    # 加载 scales
    a_scale = tl.load(a_scale_ptr + ...)
    b_scale = tl.load(b_scale_ptr + ...)
    
    # 反量化
    c = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
    
    tl.store(c_ptr + ..., c)
```

## 测试

```bash
# 运行测试
python3 int8_ops.py
```

输出示例:
```
======================================================================
INT8 Operations Test
======================================================================
Device: cuda
Triton available: True
torch._int_mm available: True

Input shape: torch.Size([2, 512, 256])
Weight shape: torch.Size([1024, 256])

--- Test INT8Quantizer ---
x_int8 dtype: torch.int8, shape: torch.Size([2, 512, 256])
x_scale shape: torch.Size([2, 512, 1])

--- Test INT8Linear ---
Output shape: torch.Size([2, 512, 1024]), dtype: torch.float16

--- Accuracy vs FP16 ---
MSE: 0.000123
Cosine Similarity: 0.999876

--- Speed Benchmark ---
INT8 time: 0.1234 ms
FP16 time: 0.2345 ms
Speedup: 1.90x
======================================================================
```

## 注意事项

1. **精度损失**: INT8 量化会有精度损失，通常 cosine similarity > 0.99
2. **速度提升**: 只有使用 Triton 或 torch._int_mm 才有速度提升
3. **显存节省**: 权重从 FP16 (2 bytes) 变为 INT8 (1 byte)，但需要额外存储 scale
4. **Triton 安装**: `pip install triton`
