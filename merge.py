from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str, required=True)
    parser.add_argument("-p", "--peft", type=str, required=True)
    parser.add_argument("-o", "--out", type=str, required=True)
    parser.add_argument("-P", "--push", action="store_true")

    parser.add_argument("-r", "--trust-remote-code", action="store_true")
    parser.add_argument("-C", "--cpu-only", action="store_true")
    parser.add_argument("-c", "--model-max-context", type=int)
    parser.add_argument("-t", "--rope-scaling-type", type=str)
    parser.add_argument("-f", "--rope-scaling-factor", type=float)

    return parser.parse_args()

def main():
    args = get_args()
    print(f"Loading base model: {args.base}")
    load_args = {}
    if not args.cpu_only:
        load_args["device_map"] = "auto"
    if isinstance(args.model_max_context, int):
        load_args['max_position_embeddings'] = args.model_max_context
        print(f"Model max context length adjusted to {args.model_max_context} tokens.")
    if isinstance(args.rope_scaling_type, str) and (isinstance(args.rope_scaling_factor, float) or isinstance(args.rope_scaling_factor, int)):
        rope_scaling_setting_dict = {"type": args.rope_scaling_type, "factor": float(args.rope_scaling_factor)}
        load_args['rope_scaling'] = rope_scaling_setting_dict
        print(f"Using rope scaling with setting: {rope_scaling_setting_dict}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        return_dict=True,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
        **load_args
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.peft, trust_remote_code=args.trust_remote_code)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=args.trust_remote_code)
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"Loading PEFT: {args.peft}")
    peft_model = PeftModel.from_pretrained(base_model, args.peft)
    print(f"Running merge_and_unload...")
    model = peft_model.merge_and_unload(progressbar=True)
    print("Saving merged model and tokenizer...")
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size='10GB')
    tokenizer.save_pretrained(args.out)
    if args.push:
        print(f"Saving to hub...")
        model.push(args.out, use_temp_dir=False)
        tokenizer.push(args.out, use_temp_dir=False)

if __name__ == "__main__" :
    main()
