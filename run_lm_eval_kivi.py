
# Import necessary modules
from utils import load_model_and_tokenizer, add_common_args
import argparse
import torch
import lm_eval
from tqdm import tqdm
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from lm_eval.utils import eval_logger as logger
import os
import json
from modules.quant_utils import Quantizer
from modules.hadamard_utils import apply_hadamard
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
from modeling_qwen_kivi import load_qwen_kivi

def run_lm_eval_zero_shot(model, tokenizer, batch_size=32, max_length=4096, task_list=["arc_easy", "hellaswag"], limit=None):
    model.seqlen = max_length
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=batch_size)
    # indexes all tasks from the lm_eval/tasks subdirectory.
    # Alternatively, you can set TaskManager(include_path="path/to/my/custom/task/configs")
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting task_manager to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in lm_eval/tasks.
    # simple_evaluate will instantiate its own task_manager is the it is set to None here.
    logger.info(f"Evaluation, Task(s): {task_list}")
    with torch.no_grad():
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_obj,
            #model_args= "add_bos_token=True" if model_type == "jamba" else "",
            tasks=task_list,
            #task_kwargs={"boolq": {"config": "default"}, "piqa": {"config": "default"}},
            task_manager=task_manager,
            log_samples=False,
            limit=limit
        ) 

    res = make_table(results)
    print(res)
    
    return results['results']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        '--tasks', type=lambda s: [item for item in s.split(',')], default=[],
        help='Task to be evaled'
    )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='batch size for lm_eval tasks'
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose information or not."
    )
    parser.add_argument(
        "--save_results",
        type=int,
        help="Whether to save the results or not."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the .json results."
    )
    parser.add_argument(
        "--use_quant",
        action="store_true"
    )
    parser.add_argument(
        "--qwen",
        action='store_true'
    )
    args = parser.parse_args()  
    logger.info("Loading model and tokenizer...")
    #model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.k_bits = 4 # current support 2/4 bit for KV Cache
    config.v_bits = 4# current support 2/4 bit for KV Cache
    config.group_size = 128
    config.residual_length = 128
    config.use_flash = True
    # the number of recent fp16 tokens
    
    if not args.qwen:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, )
            #use_fast=False, 
            #trust_remote_code=True,
            #tokenizer_type='llama') 
        print(type(tokenizer))

        model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        )
    else:
        model, tokenizer = load_qwen_kivi(
            args.model_name_or_path,
            k_bits=2,      # Key 使用 2bit 量化
            v_bits=2,      # Value 使用 2bit 量化
            group_size=128,
            residual_length=128,
            torch_dtype=torch.float16,
        )
        model.use_kivi_cache = True
    if args.use_quant:
        rank_list_k = []
        rank_list_v = []
        rank_list = torch.load(args.model_name_or_path + '/rank_list.pt')
        for k,v in rank_list.items():
            if 'k_proj' in k:
                rank_list_k.append(v)
            if 'v_proj' in k:
                rank_list_v.append(v)
        rank_sum = [rank_list_k[i] + rank_list_v[i] for i in range(len(rank_list_k))]
        tmp_ranks_sum = rank_sum
        rank_sum.sort()
        identifier = rank_sum[15]
        for name, module in model.named_modules():
            if name.endswith('v_proj') or name.endswith('k_proj'):
                layer_id = int(name.split('.')[2])
                #if tmp_ranks_sum[layer_id] > identifier:
                #    module.quantizer = Quantizer(n_bits = 16, group_size = 0, sym = True, clip_ratio = 1.0)
                #else:
                #    module.quantizer = Quantizer(n_bits = 8, group_size = 0, sym = True, clip_ratio = 1.0)
                module.fuse_hadamard()
    logger.info("Start running lm_eval zero-shot evaluation...")
    res = run_lm_eval_zero_shot(model, tokenizer, args.batch_size, task_list=args.tasks)
    
    # Create directory if it doesn't exist
    #output_dir = "./results"
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON file
    model_name = args.model_name_or_path.split("/")[-1]
    output_file = os.path.join(output_dir, f"{model_name}_{args.lt_bits}{'_had' if args.lt_hadamard else ''}.json")
    with open(output_file, "w") as f:
        json.dump(res, f, indent=4)

    print(f"Results saved to {output_file}")
    