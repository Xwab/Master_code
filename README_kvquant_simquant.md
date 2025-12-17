# KVQuant-style Simulated Quantization for LLaMA3

这个目录包含两个脚本，用于在不使用自定义CUDA kernel的情况下模拟LLaMA3（支持GQA架构）的KV cache量化。

## 脚本说明

### 1. `kvquant_simquant_ppl_eval.py` - 简化版本
- 直接使用动态量化，无需校准
- 适合快速测试和验证

### 2. `kvquant_llama3_simquant.py` - 完整版本
- 支持校准模式（收集激活值统计信息）
- 支持NUQ（非均匀量化）
- 支持保存/加载量化参数

## 使用方法

### 简单评估（动态量化，无需校准）

```bash
# 4-bit 量化评估
python3 kvquant_simquant_ppl_eval.py meta-llama/Llama-3-8B \
    --abits 4 \
    --datasets wikitext2 \
    --seqlen 2048

# 4-bit + 1% outliers (dense-and-sparse)
python3 kvquant_simquant_ppl_eval.py meta-llama/Llama-3-8B \
    --abits 4 \
    --include_sparse \
    --sparsity_threshold 0.99 \
    --datasets wikitext2

# 2-bit 量化 + attention sink
python3 kvquant_simquant_ppl_eval.py meta-llama/Llama-3-8B \
    --abits 2 \
    --first_few_fp16 1 \
    --datasets wikitext2

# FP16 基线（无量化）
python3 kvquant_simquant_ppl_eval.py meta-llama/Llama-3-8B \
    --abits 16 \
    --datasets wikitext2
```

### 带校准的评估（更精确）

```bash
# Step 1: 校准 - 收集激活值统计并保存量化参数
python3 kvquant_llama3_simquant.py meta-llama/Llama-3-8B \
    --calibrate \
    --abits 4 \
    --nsamples 16 \
    --seqlen 2048 \
    --include_sparse \
    --sparsity_threshold 0.99 \
    --quantizer_path quantizers_llama3_4bit.pkl

# Step 2: 使用校准参数进行评估
python3 kvquant_llama3_simquant.py meta-llama/Llama-3-8B \
    --abits 4 \
    --include_sparse \
    --sparsity_threshold 0.99 \
    --quantizer_path quantizers_llama3_4bit.pkl \
    --datasets wikitext2
```

### 使用NUQ（非均匀量化）

```bash
# 校准并保存NUQ参数
python3 kvquant_llama3_simquant.py meta-llama/Llama-3-8B \
    --calibrate \
    --abits 4 \
    --nuq \
    --include_sparse \
    --sparsity_threshold 0.99 \
    --quantizer_path quantizers_nuq4.pkl

# 评估
python3 kvquant_llama3_simquant.py meta-llama/Llama-3-8B \
    --abits 4 \
    --nuq \
    --include_sparse \
    --sparsity_threshold 0.99 \
    --quantizer_path quantizers_nuq4.pkl \
    --datasets wikitext2
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--abits` | 量化位宽 (2, 3, 4, 8, 16) | 4 |
| `--perchannel` | 使用per-channel量化的层 | ["k_proj"] |
| `--pertoken` | 使用per-token量化的层 | ["v_proj"] |
| `--include_sparse` | 使用dense-and-sparse量化 | False |
| `--sparsity_threshold` | 异常值阈值百分位 | 0.99 |
| `--nuq` | 使用非均匀量化 | False |
| `--first_few_fp16` | 保持FP16的前N个token（attention sink）| -1 |
| `--datasets` | 评估数据集（逗号分隔）| wikitext2 |
| `--seqlen` | 序列长度 | 2048 |
| `--limit` | 最大评估样本数 | 512 |

## 量化策略说明

KVQuant使用以下量化策略：

1. **K (Key) - Per-channel量化**: 沿着head dimension方向进行量化，每个channel共享一个scale
2. **V (Value) - Per-token量化**: 沿着token方向进行量化，每个token共享一个scale

这种策略对GQA（LLaMA3使用）是兼容的，因为：
- 量化是在`k_proj`和`v_proj`的**输出**上进行的
- 与KV heads数量解耦

## 与KVQuant官方实现的区别

| 特性 | 官方KVQuant | 本实现 |
|------|------------|--------|
| 自定义CUDA kernel | 需要 | 不需要 |
| 实际加速 | ✅ 有 | ❌ 无（仅模拟） |
| 精度评估 | ✅ 支持 | ✅ 支持 |
| Fisher信息 | ✅ 支持 | ❌ 不支持 |
| GQA支持 | 部分 | ✅ 完整 |

## 注意事项

1. **模拟量化不会带来实际的推理加速**，仅用于评估量化精度损失
2. 如果需要实际加速，请使用KVQuant官方的CUDA kernel实现
3. 首次运行会自动下载数据集，可能需要一些时间
4. 建议使用flash attention 2以节省显存
