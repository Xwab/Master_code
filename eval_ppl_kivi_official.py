"""
使用 KIVI 官方的 PPL 测试方法

KIVI 仓库通常包含 eval_long_ppl.py 或类似脚本
这里提供一个封装，兼容你的 cache_testloader
"""

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

# 添加 KIVI 路径
KIVI_PATH = "/root/KIVI"
if KIVI_PATH not in sys.path:
    sys.path.insert(0, KIVI_PATH)


def get_ppl_eval_loaders(dataset_name, tokenizer, seqlen=2048):
    """加载 PPL 评估数据集"""
    from datasets import load_dataset
    
    if dataset_name == "wikitext2":
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    elif dataset_name == "c4":
        testdata = load_dataset(
            'allenai/c4', 'allenai--c4', 
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
            split='validation'
        )
        testenc = tokenizer(' '.join(testdata[:1100]['text']), return_tensors='pt')
    elif dataset_name == "ptb":
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return testenc


@torch.no_grad()
def eval_ppl_kivi_official(
    model,
    tokenizer,
    model_name: str,
    dataset: str = "wikitext2",
    seqlen: int = 2048,
    device: str = "cuda",
    limit: int = None,
):
    """
    使用 KIVI 官方的方式测试 PPL
    
    这个实现模拟 KIVI 官方的 eval_long_ppl.py 逻辑
    """
    model = model.to(device)
    model.eval()
    
    # 加载或使用缓存的 testloader
    cache_testloader = f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
    
    if os.path.exists(cache_testloader):
        print(f"Loading cached testloader from {cache_testloader}")
        testloader = torch.load(cache_testloader)
    else:
        print(f"Creating testloader for {dataset}...")
        testloader = get_ppl_eval_loaders(dataset, tokenizer, seqlen)
        torch.save(testloader, cache_testloader)
        print(f"Saved to {cache_testloader}")
    
    testenc = testloader.input_ids
    nsamples = testenc.numel() // seqlen
    
    if limit is not None:
        nsamples = min(nsamples, limit)
    
    print(f"Evaluating {nsamples} samples, seqlen={seqlen}")
    
    # 获取 KIVI 参数
    k_bits = getattr(model.config, 'k_bits', 2)
    v_bits = getattr(model.config, 'v_bits', 2)
    group_size = getattr(model.config, 'group_size', 128)
    residual_length = getattr(model.config, 'residual_length', 128)
    
    print(f"KIVI config: k_bits={k_bits}, v_bits={v_bits}, "
          f"group_size={group_size}, residual_length={residual_length}")
    
    nlls = []
    
    for i in tqdm(range(nsamples), desc="Evaluating PPL"):
        batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(device)
        
        # KIVI 官方方式：使用 model.generate 的内部逻辑
        # 或者直接调用 model forward with use_cache=True
        
        # 方法：一次性 forward 整个序列（KIVI 会在 attention 中自动处理）
        outputs = model(
            batch,
            use_cache=True,
        )
        
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        neg_log_likelihood = loss.float() * (seqlen - 1)
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
    
    print(f"\n{dataset} PPL: {ppl.item():.4f}")
    
    return ppl.item()


# ============================================================
# 方法 2: 直接调用 KIVI 的 eval 脚本（如果存在）
# ============================================================

def run_kivi_eval_script(
    model_path: str,
    dataset: str = "wikitext2",
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 128,
    residual_length: int = 128,
    seqlen: int = 2048,
    num_samples: int = None,
):
    """
    直接运行 KIVI 官方的 eval 脚本
    
    使用方法:
        run_kivi_eval_script(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            k_bits=2,
            v_bits=2,
        )
    """
    import subprocess
    
    # 构建命令
    cmd = [
        "python", f"{KIVI_PATH}/eval_long_ppl.py",
        "--model_name_or_path", model_path,
        "--k_bits", str(k_bits),
        "--v_bits", str(v_bits),
        "--group_size", str(group_size),
        "--residual_length", str(residual_length),
        "--dataset", dataset,
        "--seqlen", str(seqlen),
    ]
    
    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result


# ============================================================
# 方法 3: 使用你的 cache_testloader 与 KIVI 结合
# ============================================================

@torch.no_grad()
def eval_ppl_with_cached_loader(
    model,
    cache_testloader_path: str,
    seqlen: int = 2048,
    device: str = "cuda",
    limit: int = None,
):
    """
    使用预先缓存的 testloader 文件
    
    Args:
        model: KIVI 模型
        cache_testloader_path: 缓存文件路径，如 
            "/tmp/wikitext2_testloader_meta-llama_Llama-3.1-8B-Instruct_all.cache"
        seqlen: 序列长度
        device: 设备
        limit: 最大样本数
    """
    model = model.to(device)
    model.eval()
    
    # 加载缓存的 testloader
    if not os.path.exists(cache_testloader_path):
        raise FileNotFoundError(f"Cache file not found: {cache_testloader_path}")
    
    print(f"Loading testloader from {cache_testloader_path}")
    testloader = torch.load(cache_testloader_path)
    
    testenc = testloader.input_ids
    nsamples = testenc.numel() // seqlen
    
    if limit is not None:
        nsamples = min(nsamples, limit)
    
    print(f"Total tokens: {testenc.numel()}, samples: {nsamples}, seqlen: {seqlen}")
    
    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    
    for i in tqdm(range(nsamples), desc="Evaluating"):
        batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(device)
        
        outputs = model(batch, use_cache=True)
        logits = outputs.logits
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        nlls.append(loss.float() * (seqlen - 1))
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
    
    print(f"PPL: {ppl.item():.4f}")
    return ppl.item()


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("""
使用示例:

1. 使用官方风格的 PPL 测试:

    from transformers import AutoTokenizer
    from KIVI.models.llama_kivi import LlamaForCausalLM_KIVI
    
    # 加载模型
    model = LlamaForCausalLM_KIVI.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        k_bits=2,
        v_bits=2,
        group_size=128,
        residual_length=128,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    # 测试 PPL
    from eval_ppl_kivi_official import eval_ppl_kivi_official
    
    ppl = eval_ppl_kivi_official(
        model=model,
        tokenizer=tokenizer,
        model_name="llama3.1-8b",
        dataset="wikitext2",
        seqlen=2048,
        limit=100,
    )

2. 使用已缓存的 testloader:

    from eval_ppl_kivi_official import eval_ppl_with_cached_loader
    
    cache_path = "/tmp/wikitext2_testloader_meta-llama_Llama-3.1-8B-Instruct_all.cache"
    
    ppl = eval_ppl_with_cached_loader(
        model=model,
        cache_testloader_path=cache_path,
        seqlen=2048,
        limit=100,
    )

3. 直接运行 KIVI 官方脚本:

    from eval_ppl_kivi_official import run_kivi_eval_script
    
    run_kivi_eval_script(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        k_bits=2,
        v_bits=2,
        dataset="wikitext2",
    )
""")
