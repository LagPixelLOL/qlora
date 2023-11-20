from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str)
    parser.add_argument("--peft", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--push", action="store_true")

    parser.add_argument("-c", "--model_max_context", type=int)
    parser.add_argument("-t", "--rope_scaling_type", type=str)
    parser.add_argument("-f", "--rope_scaling_factor", type=float)

    return parser.parse_args()

def main():
    args = get_args()
    print(f"Loading base model: {args.base}")
    load_args = {}
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
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **load_args
    )
    print(f"Loading PEFT: {args.peft}")
    model = PeftModel.from_pretrained(base_model, args.peft)
    print(f"Running merge_and_unload...")
    model = model.merge_and_unload(progressbar=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size='10GB')
    tokenizer.save_pretrained(args.out)
    if args.push:
        print(f"Saving to hub...")
        model.push(args.out, use_temp_dir=False)
        tokenizer.push(args.out, use_temp_dir=False)

if __name__ == "__main__" :
    main()