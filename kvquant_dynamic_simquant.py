"""
KVQuant Dynamic SimQuant - 简化版本，无需校准集

所有量化参数在 forward 时动态计算：
- 动态计算 min/max 范围
- 动态检测 outliers
- 支持 per-channel (K) 和 per-token (V) 量化
"""

import torch
import torch.nn as nn
import argparse
import logging
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 动态量化函数
# ============================================================================

def dynamic_quantize(x, bits, qchannel=-1, include_sparse=False, sparsity_threshold=0.99):
    """
    动态量化：在 forward 时计算所有参数
    
    Args:
        x: 输入 tensor
        bits: 量化位数
        qchannel: 量化维度 (0=per-channel, -1=per-token)
        include_sparse: 是否使用稀疏量化 (outliers 保持 FP16)
        sparsity_threshold: outlier 阈值 (如 0.99 = 1% outliers)
    """
    if bits >= 16:
        return x
    
    x_float = x.float()
    
    # 检测 outliers
    outlier_mask = None
    if include_sparse:
        t = 1 - ((1 - sparsity_threshold) / 2)
        upper = torch.quantile(x_float, t, dim=qchannel, keepdim=True)
        lower = torch.quantile(x_float, 1 - t, dim=qchannel, keepdim=True)
        outlier_mask = (x_float > upper) | (x_float < lower)
    
    # 计算量化范围
    if include_sparse and outlier_mask is not None:
        # 排除 outliers 计算范围
        x_masked = x_float.clone()
        x_masked[outlier_mask] = 0
        non_outlier_count = (~outlier_mask).sum(dim=qchannel, keepdim=True).clamp(min=1)
        x_sum = x_masked.sum(dim=qchannel, keepdim=True)
        x_mean = x_sum / non_outlier_count
        x_masked[outlier_mask] = x_mean.expand_as(x_masked)[outlier_mask]
        maxval = x_masked.max(dim=qchannel, keepdim=True).values
        minval = x_masked.min(dim=qchannel, keepdim=True).values
    else:
        maxval = x_float.max(dim=qchannel, keepdim=True).values
        minval = x_float.min(dim=qchannel, keepdim=True).values
    
    # 量化
    scale = (2**bits - 1) / (maxval - minval).clamp(min=1e-8)
    zero_point = minval * scale
    
    qx = torch.round(scale * x_float - zero_point).clamp(0, 2**bits - 1)
    x_dequant = (qx + zero_point) / scale
    
    # 恢复 outliers
    if include_sparse and outlier_mask is not None:
        x_dequant[outlier_mask] = x_float[outlier_mask]
    
    return x_dequant.to(x.dtype)


# ============================================================================
# QuantLinearSim - 动态量化版本
# ============================================================================

class QuantLinearSimDynamic(nn.Module):
    """
    动态模拟量化层 - 无需校准
    """
    
    def __init__(self, original_linear, bits, perchannel=True, 
                 include_sparse=False, sparsity_threshold=0.99):
        super().__init__()
        
        self.bits = bits
        self.qchannel = 0 if perchannel else -1
        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # 复制权重，保持原设备和类型
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype
        
        # 存储转置的权重 (用于 x @ W 计算)
        self.weight = nn.Parameter(
            original_linear.weight.T.detach().clone(),
            requires_grad=False
        )
        
        if original_linear.bias is not None:
            self.bias = nn.Parameter(
                original_linear.bias.detach().clone(),
                requires_grad=False
            )
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.view(-1, x.shape[-1])
        
        # 确保在同一设备
        device = self.weight.device
        dtype = self.weight.dtype
        x_flat = x_flat.to(device=device, dtype=dtype)
        
        # 线性计算
        y = x_flat @ self.weight
        if self.bias is not None:
            y = y + self.bias
        
        # 动态量化
        y = dynamic_quantize(
            y, 
            bits=self.bits,
            qchannel=self.qchannel,
            include_sparse=self.include_sparse,
            sparsity_threshold=self.sparsity_threshold
        )
        
        return y.view(out_shape)
    
    def extra_repr(self):
        return f'in={self.in_features}, out={self.out_features}, bits={self.bits}, sparse={self.include_sparse}'


# ============================================================================
# 模型修改函数
# ============================================================================

def replace_with_dynamic_quant(model, bits, 
                                k_perchannel=True, v_perchannel=False,
                                include_sparse=False, sparsity_threshold=0.99,
                                k_match=["k_proj"], v_match=["v_proj"]):
    """
    将模型中的 k_proj/v_proj 替换为动态量化版本
    """
    replaced = 0
    
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        
        # 检查是否匹配 k_proj 或 v_proj
        is_k = any(k in name for k in k_match)
        is_v = any(v in name for v in v_match)
        
        if not (is_k or is_v):
            continue
        
        # 确定量化方式
        perchannel = k_perchannel if is_k else v_perchannel
        
        # 创建量化层
        quant_layer = QuantLinearSimDynamic(
            module, 
            bits=bits,
            perchannel=perchannel,
            include_sparse=include_sparse,
            sparsity_threshold=sparsity_threshold
        )
        
        # 替换
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        
        setattr(parent, child_name, quant_layer)
        replaced += 1
        logger.debug(f"Replaced {name} with {bits}-bit dynamic quant (perchannel={perchannel})")
    
    logger.info(f"Replaced {replaced} layers with dynamic quantization")
    return model


# ============================================================================
# PPL 评估
# ============================================================================

@torch.no_grad()
def evaluate_ppl(model, tokenizer, dataset='wikitext2', seqlen=2048, device='cuda'):
    """评估 perplexity"""
    logger.info(f"Evaluating PPL on {dataset}...")
    
    if dataset == 'wikitext2':
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    testenc = testenc.input_ids.to(device)
    nsamples = testenc.numel() // seqlen
    
    model.eval()
    nlls = []
    
    for i in tqdm(range(nsamples), desc="Evaluating"):
        batch = testenc[:, i*seqlen:(i+1)*seqlen]
        
        with torch.cuda.amp.autocast():
            outputs = model(batch)
            logits = outputs.logits
        
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * seqlen)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()


# ============================================================================
# lm-eval 集成
# ============================================================================

def run_lm_eval(model, tokenizer, tasks, batch_size=8, device='cuda'):
    """运行 lm-eval 评估"""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error("lm-eval not installed. Run: pip install lm-eval")
        return None
    
    logger.info(f"Running lm-eval on tasks: {tasks}")
    
    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        log_samples=False,
    )
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='KVQuant Dynamic SimQuant')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    
    # 量化参数
    parser.add_argument('--bits', type=int, default=4, choices=[2, 3, 4, 8, 16],
                        help='Quantization bits')
    parser.add_argument('--include_sparse', action='store_true',
                        help='Use sparse quantization (outliers in FP16)')
    parser.add_argument('--sparsity_threshold', type=float, default=0.99,
                        help='Outlier threshold (0.99 = 1%% outliers)')
    
    # 评估参数
    parser.add_argument('--eval_ppl', action='store_true', help='Evaluate PPL')
    parser.add_argument('--tasks', type=str, default='',
                        help='lm-eval tasks (comma-separated)')
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("KVQuant Dynamic SimQuant (No Calibration)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Bits: {args.bits}")
    logger.info(f"Sparse: {args.include_sparse} (threshold={args.sparsity_threshold})")
    logger.info("=" * 60)
    
    # 加载模型
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    model.eval()
    
    # 应用动态量化
    if args.bits < 16:
        logger.info(f"Applying {args.bits}-bit dynamic quantization...")
        model = replace_with_dynamic_quant(
            model,
            bits=args.bits,
            k_perchannel=True,   # K: per-channel
            v_perchannel=False,  # V: per-token
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold
        )
    
    # 评估 PPL
    if args.eval_ppl:
        ppl = evaluate_ppl(model, tokenizer, seqlen=args.seqlen, device=device)
        logger.info(f"Perplexity: {ppl:.4f}")
    
    # 运行 lm-eval
    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(',')]
        results = run_lm_eval(model, tokenizer, task_list, args.batch_size, device)
        if results:
            from lm_eval.utils import make_table
            print(make_table(results))


if __name__ == '__main__':
    main()
