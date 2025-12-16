import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
from utils import load_model_and_tokenizer, add_common_args
from loguru import logger
from modules.quant_utils import Quantizer
from modules.hadamard_utils import apply_hadamard
def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    elif "c4" in name:
        # Wrapper for tokenized input IDs
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
                
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        #testenc = tokenizer("\n\n".join(valdata["text"]), return_tensors="pt")
        testenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        testenc = TokenizerWrapper(testenc)
        return testenc
    elif "ptb" in name:
        valdata = load_dataset(
            "ptb-text-only/ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    else:
        raise NotImplementedError

@torch.no_grad()
def eval_ppl(model, tokenizer, model_name, datasets, seqlen=2048, device="cuda",limit=512):
    model = model.to(device)
    if isinstance(device, str):
        device = torch.device(device)

    results = {}

    for dataset in datasets.split(","):
        cache_testloader = (
            f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
        )
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
        else:
            testloader = get_ppl_eval_loaders(dataset, tokenizer)
            torch.save(testloader, cache_testloader)
        
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        nlls = []

        for i in tqdm(range(nsamples)):
            if i >= limit:
                break
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(
                    device
                )
            ### 这里是新加的更新scaling matrix
            #scaling_matrix_list = attn_output_calib_for_rank_search_v2_for_update(model, batch)
            #for name, module in model.named_modules():
            #    if 'k_proj' in name:
            #        module.update_scaling_matrix(
            #            scaling_matrix_list[name.replace("k_proj", "in_scale")]
            #        )
            #    if 'v_proj' in name:
            #        module.update_scaling_matrix(
            #            scaling_matrix_list[name.replace("v_proj", "in_scale_x")]
            #        )
            ###

            outputs = model.model(batch)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)  # .contiguous()
            shift_logits = logits[:, :-1, :]  # .contiguous()
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
            
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
        model.config.use_cache = use_cache
        results.update({dataset: ppl.item()})

    return results
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--datasets', type=str, help='datasets to evaluate', default='wikitext2, ptb')
    parser.add_argument('--seqlen', type=int, help='sequence length for ppl evaluation', default=2048)
    parser.add_argument("--device", type=str, help="device to run the model on", default="cuda")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information or not.")
    parser.add_argument(
        "--use_quant",
        action="store_true"
    )
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
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
    logger.info(f"Start evaluating ppl...")
    logger.info(f"*model: {args.model_name_or_path}")
    logger.info(f"*datasets: {args.datasets}")
    logger.info(f"*sequence length {args.seqlen}")
    results = eval_ppl(model, tokenizer, args.model_name_or_path, args.datasets, args.seqlen, args.device)
    for dataset, ppl in results.items():
        logger.info(f"PPL: {ppl}")