#Adapted from https://github.com/THUDM/LongBench/blob/main/pred.py
import os
from datasets import load_dataset
import torch
from loguru import logger
from tqdm import tqdm
import numpy as np
import random
import argparse
import time
import json
from loguru import logger
from datetime import datetime
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
os.environ["WANDB_DISABLED"] = "true"
from modeling_qwen_kivi import load_qwen_kivi
from modeling_llama_kivi import load_llama_kivi

from longbench_utils import scorer, MODEL2MAXLEN, DATASET2PROMPT, DATASET2MAXLEN
from utils import load_model_and_tokenizer, add_common_args
from modules.kivi_cache_general import create_kivi_cache
import warnings
warnings.filterwarnings("ignore", 
                        message=".*`do_sample` is set to `False`. However, `top_p` is set to.*",
                        category=UserWarning)

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # Copy from KIVI
    if "longchat" in model_name.lower() or "vicuna" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "llama" in model_name.lower() or 'qwen2' in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, limit=2048):
    preds = []
    i = 0
    for json_obj in tqdm(data):
        if i >= limit:
            break
        i += 1
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        #build mixed precision cache
        

        '''kivi_cache = create_kivi_cache(
            k_bits= 2,
            v_bits= 2,
            group_size=128,
            residual_length=128,
        )'''

        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                use_cache=True,
                #past_key_values=kivi_cache,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                use_cache=True,
                #past_key_values=kivi_cache,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
        torch.cuda.empty_cache()
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main(args):
    model2maxlen = MODEL2MAXLEN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.k_bits = 4 # current support 2/4 bit for KV Cache
    config.v_bits = 4# current support 2/4 bit for KV Cache
    config.group_size = 128
    config.residual_length = 128
    config.use_flash = True
    # the number of recent fp16 tokens
    if not args.qwen:
        model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, )
        #use_fast=False, 
        #trust_remote_code=True, )
        #tokenizer_type='llama') 
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
                # = int(name.split('.')[2])
                #if tmp_ranks_sum[layer_id] > identifier:
                #    #module.quantizer = Quantizer(n_bits = 16, group_size = 0, sym = True, clip_ratio = 1.0)
                #    module.quantizer = Quantizer(n_bits = 2, group_size = 4, sym = True, clip_ratio = 1.0)
                #else:
                    #module.quantizer = Quantizer(n_bits = 8, group_size = 0, sym = True, clip_ratio = 1.0)
                #    module.quantizer = Quantizer(n_bits = 2, group_size = 4, sym = True, clip_ratio = 1.0)
                #if layer_id == 0:
                #    module.quantizer = Quantizer(n_bits = 2, group_size = 0, sym = True, clip_ratio = 1.0)
                module.fuse_hadamard()

    #NOTE(brian1009): This is a hack to get the model name
    # We assume the model name is the inside the last part of the path
    # and the Palu's compression information is follow by the model name with a "_"
    # Hence, we split the path by "/" and then keep only the first part by "_"
    # Example: Mistral-7B-Instruct-v0.2_ratio-0.7_gs-4-fisher_uniform
    raw_model_name = args.model_name_or_path.split("/")[-1]
    model_type = model.config.model_type
    if 'llama' in model_type.lower():
        print("it is llama")
        tokenizer.padding_side = "left"
        
    model.eval()

    if not model_type in model2maxlen:
        raise ValueError(f"Model {model_type} not supported")
    
    max_length = model2maxlen[model_type]
    if not args.qwen:
        max_length = 24000
    else:
        max_length = 32000
    logger.info(f"Running model: {raw_model_name}")
    logger.info(f"Max length: {max_length}")
    datasets = args.datasets
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = DATASET2PROMPT
    dataset2maxlen = DATASET2MAXLEN
    # predict on each dataset
    if not os.path.exists("Longbench/pred"):
        os.makedirs("Longbench/pred")
    
    results = {}
    
    for dataset in datasets:
        logger.info("Evaluating dataset: {}".format(dataset))
        start_time = time.time()
        data = load_dataset('zai-org/LongBench', dataset, split='test')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_type)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time for dataset {dataset}: {elapsed_time/60} minutes")
        
        # calculate score
        predictions, answers, lengths = [], [], []
        for pred in preds:
            predictions.append(pred["pred"])
            answers.append(pred["answers"])
            if "length" in pred:
                lengths.append(pred["length"])
            all_classes = pred["all_classes"]
        score = scorer(dataset, predictions, answers, all_classes)
        logger.info(f"dataset: {dataset}")
        logger.info(f"score: {score}")

        # Log the results of each datasets
        with open(f"results/Longbench/{raw_model_name}_bits_{args.lt_bits}.json", "a") as f:
            data_to_log = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": dataset,
                "score": score,
            }
            json.dump(data_to_log, f)
            f.write("\n")
    
if __name__ == '__main__':
    seed_everything(42)    
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        '--datasets', type=lambda s: [item for item in s.split(',')],
        #default=["passage_count", "lcc","narrativeqa","multifieldqa_en","multifieldqa_zh","hotpotqa","musique","dureader",
        #          "gov_report","multi_news","vcsum","trec","samsum","lsht",
        #          "passage_retrieval_en","passage_retrieval_zh","repobench-p"],
        #default=["qasper", "2wikimqa","qmsum","triviaqa","passage_count", "lcc"],
        default=['hotpotqa', 'musique', 'dureader'],
        help='The datasets to be evaluated'
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose information or not."
    )
    parser.add_argument(
        "--use_quant",
        action="store_true"
    )
    parser.add_argument(
        "--qwen",
        action="store_true"
    )
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    #Create directory to log evaluation results.
    os.makedirs("results/Longbench", exist_ok=True)
    
    main(args)
    
    