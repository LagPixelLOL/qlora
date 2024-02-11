from collections import defaultdict
import copy
import json
import os
import shutil
from dataclasses import dataclass, field
import dataclasses
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
from torch.nn.utils.rnn import pad_sequence
import argparse
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from datasets import load_dataset, Dataset, DatasetDict
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from accelerate import Accelerator, DistributedType

def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="owolewd",
        metadata={"help": "The model name in huggingface or the path to the local model folder."}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM.from_pretrained."}
    )
    model_max_context: Optional[int] = field(
        default=None,
        metadata={"help": "Set the max context length of the model, will default to the length from model config."}
    )
    rope_scaling_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "The type of rope scaling, will default to NO rope scaling. You can use either \"linear\" or \"dynamic\", dynamic scaling is better than linear. "
                    "NOTE: It's only available in a few models[LLaMA, Falcon, OpenLLaMA, GPT-NeoX, Fuyu, Phi, Persimmon], "
                    "you CAN'T use it in other models, please check the transformers library document for more information."
        }
    )
    rope_scaling_factor: Optional[float] = field(
        default=None,
        metadata={"help": "The scaling factor, must be > 1, the final model context length will be <model_max_context * rope_scaling_factor>, will default to no scaling."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024,
        metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."},
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded(And possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded(And possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used[alpaca, chip2, self-instruct, hh-rlhf]."}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "If you want to use custom huggingface cache dir or not, will default to the default huggingface cache dir."}
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Fine-tune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA R dimension(LoRA rank)."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "LoRA alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=696969,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints.'})
    save_as_safetensors: bool = field(default=True, metadata={"help": 'Save the trained model in safetensors format.'})
    max_shard_size: str = field(default="10GB", metadata={"help": "Max shard size when saving model after full finetune."})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used.'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step.'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take.'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW.'}) # Use lora dropout instead for regularization if needed.
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    no_eval: bool = field(default=False, metadata={"help": 'When passed, disable eval.'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis.'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for.'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss.'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints.'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model.'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten.'})
    use_flash_attention_2: bool = field(default=True, metadata={"help": 'Use flash attention 2 to load the model.'})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

def is_deepspeed_zero_3(accelerator):
    state = accelerator.state
    return state.distributed_type == DistributedType.DEEPSPEED and state.deepspeed_plugin.deepspeed_config['zero_optimization']['stage'] == 3

def find_all_linear_names(args, model):
    nn_class = bnb.nn.Linear4bit if args.bits == 4 else bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn_class):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # Needed for 16 bits
        lora_module_names.remove('lm_head')
    if isinstance(model, transformers.MixtralForCausalLM):
        lora_module_names.remove('w1'); lora_module_names.remove('w2'); lora_module_names.remove('w3')
    return list(lora_module_names)

class SavePeftModelCallback(transformers.TrainerCallback):

    def __init__(self, trainer, **_):
        self.trainer = trainer

    def save_model(self, args, state, kwargs):
        accelerator = self.trainer.accelerator
        accelerator.wait_for_everyone()
        accelerator.print('Saving PEFT checkpoint...')
        checkpoint_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_dir = os.path.join(checkpoint_dir, "adapter_model")

        adapter_model_safetensors_file = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        adapter_model_safetensors_is_file = os.path.isfile(adapter_model_safetensors_file)
        adapter_model_bin_file = os.path.join(checkpoint_dir, "adapter_model.bin")
        adapter_config_file = os.path.join(checkpoint_dir, "adapter_config.json")

        accelerator.wait_for_everyone()
        if (adapter_model_safetensors_is_file or os.path.isfile(adapter_model_bin_file)) and os.path.isfile(adapter_config_file):
            if accelerator.is_main_process:
                try:
                    print("PEFT checkpoint already saved by the trainer, moving it to the target directory...")
                    os.makedirs(peft_model_dir, exist_ok=True)
                    shutil.move(adapter_config_file, peft_model_dir)
                    if adapter_model_safetensors_is_file:
                        shutil.move(adapter_model_safetensors_file, peft_model_dir)
                    else:
                        shutil.move(adapter_model_bin_file, peft_model_dir)
                except Exception as e:
                    print(f"Error occurred while moving the adapter model or adapter config: {e}")
        elif getattr(self.trainer, "deepspeed"):
            accelerator.print('>>>>> DeepSpeed saving... <<<<<')
            state_dict = accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = accelerator.unwrap_model(self.trainer.deepspeed)
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained(peft_model_dir, state_dict=state_dict, safe_serialization=args.save_as_safetensors)
        else:
            kwargs["model"].save_pretrained(peft_model_dir, safe_serialization=args.save_as_safetensors)

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            pytorch_model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            try:
                if os.path.exists(os.path.join(checkpoint_dir, f'global_step{state.global_step}')):
                    print(f'Cleaning up global_step{state.global_step}...')
                    shutil.rmtree(os.path.join(checkpoint_dir, f'global_step{state.global_step}'))
            except Exception as e:
                print(f'Failed to clean up global_step{state.global_step}: {e}')
        return checkpoint_dir

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        accelerator = self.trainer.accelerator
        checkpoint_dir = self.save_model(args, state, kwargs)
        if accelerator.is_main_process:
            try:
                shutil.move(checkpoint_dir, os.path.join(args.output_dir, "final"))
            except Exception as e:
                print(f"Error occurred while moving the final output to the target directory: {e}")
        if accelerator.is_local_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            fname = os.path.join(args.output_dir, 'completed')
            with open(fname, "a", encoding="utf8"):
                os.utime(fname, None)

def get_accelerate_model(args, checkpoint_dir, accelerator):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    elif is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
    else:
        raise AssertionError("You must have 1 or more GPUs to use this script.")

    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    if not is_deepspeed_zero_3(accelerator):
        device_map = "auto"

    # If we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if not is_deepspeed_zero_3(accelerator):
            device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    if compute_dtype == torch.float16 and is_ipex_available() and torch.xpu.is_available():
        compute_dtype = torch.bfloat16
        accelerator.print('Intel XPU does not support float16 yet, so switched to bfloat16.')
    if compute_dtype == torch.float16:
        if torch.cuda.is_bf16_supported():
            accelerator.print('=' * 80)
            accelerator.print('Your GPU supports bfloat16, you can accelerate training with it by passing argument `--bf16`.')
            accelerator.print('=' * 80)

    accelerator.print(f'Loading base model {args.model_name_or_path}...')

    load_args = {}
    if isinstance(args.model_max_context, int):
        load_args['max_position_embeddings'] = args.model_max_context
        accelerator.print(f"Model max context length adjusted to {args.model_max_context} tokens.")
    if isinstance(args.rope_scaling_type, str) and (isinstance(args.rope_scaling_factor, float) or isinstance(args.rope_scaling_factor, int)):
        rope_scaling_setting_dict = {"type": args.rope_scaling_type, "factor": float(args.rope_scaling_factor)}
        load_args['rope_scaling'] = rope_scaling_setting_dict
        accelerator.print(f"Using rope scaling with setting: {rope_scaling_setting_dict}")

    if args.bits in [4, 8]:
        accelerator.print(f"Using {args.bits} bits quantization.")
        load_args['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        )
    else:
        accelerator.print(f"Using {args.bits} bits.")
    accelerator.print(f"Using compute dtype {compute_dtype}.")

    if not is_deepspeed_zero_3(accelerator):
        load_args['device_map'] = device_map

    if args.use_flash_attention_2:
        load_args["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        max_memory=max_memory,
        torch_dtype=compute_dtype,
        trust_remote_code=args.trust_remote_code,
        **load_args
    )

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = compute_dtype

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        trust_remote_code=args.trust_remote_code,
        legacy=False,
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
            accelerator.print("Pad token(pad) doesn't exist in this model, so it's set to unknown(unk) token.")
        else:
            add_special_tokens_smart({"pad_token": "[PAD]"}, tokenizer, model, accelerator)
            accelerator.print("Pad token(pad) doesn't exist in this model, so it's added.")
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
        accelerator.print("Before of string(bos) token doesn't exist in this model, so it's set to end of string(eos) token.")

    if not args.full_finetune and args.bits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            accelerator.print("Loading adapters from checkpoint...")
            model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            accelerator.print(f'Adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.enable_input_require_grads()
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) or 'norm' in name or ('lm_head' in name or 'embed_tokens' in name) and hasattr(module, 'weight'):
            module.to(compute_dtype)
    return model, tokenizer

def add_special_tokens_smart(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    accelerator: Accelerator,
):
    """
    Add new tokens, then resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        accelerator.print(f"Added {num_new_tokens} new special token{'s' if num_new_tokens != 1 else ''}, resizing token embeddings...")
        model.resize_token_embeddings(len(tokenizer))

def print_trainable_parameters(args, model, accelerator):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    accelerator.print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    return DatasetDict({'train': full_dataset})

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args, accelerator) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}.")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} isn't implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'input-output':
            pass # leave as is.
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            accelerator.print('Splitting train dataset to train and validation according to `eval_dataset_size`...')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )

def get_last_checkpoint(checkpoint_dir, accelerator):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith('checkpoint-'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = os.path.join(checkpoint_dir, f'checkpoint-{max_step}')
        accelerator.print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def train():
    hfparser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments))
    model_args, data_args, training_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Args checks
    assert not extra_args, "You passed extra args which are not used, please remove these args: " + str(extra_args)
    assert training_args.bits in [4, 8, 16, 32], f"Invalid bits value \"{training_args.bits}\", please use one of [4, 8, 16, 32]."
    accelerator = Accelerator()
    if is_deepspeed_zero_3(accelerator) and training_args.bits not in [16, 32]:
        training_args.bits = 16
        accelerator.print("You can't use 4 or 8 bits when training with DeepSpeed ZeRO stage 3, automatically set bits to 16.")

    # Replace generation config.
    training_args = dataclasses.replace(training_args, generation_config=transformers.GenerationConfig(**vars(generation_args)))
    # Need to set remove_unused_columns to False for the (Seq2Seq)Trainer to not delete columns.
    training_args.remove_unused_columns = False
    training_args.do_eval = not training_args.no_eval
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))

    # Args checks again.
    # Accelerator needs to be re-initialized after training args re-init
    # (At both hfparser.parse_args_into_dataclasses and dataclasses.replace) for some reason or the state object will be broken.
    accelerator = Accelerator()
    if is_deepspeed_zero_3(accelerator) and (args.bf16 or args.fp16):
        assert accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']['stage3_gather_16bit_weights_on_model_save'], \
        "You are using (b)float16 training with DeepSpeed ZeRO stage 3, but you didn't allow 16 bits weights gathering, please pass `--zero3_save_16bit_model True` to `accelerate launch`."

    if args.full_finetune:
        assert args.bits in [16, 32], "You are doing full finetune but you are not using 16 or 32 bits."

    rope_scaling_valid_types = ['linear', 'dynamic']
    if args.rope_scaling_type is not None or args.rope_scaling_factor is not None:
        assert args.rope_scaling_type is not None, "You have rope scaling factor set but you didn't set a rope scaling type, please use one of " + str(rope_scaling_valid_types) + "."
        assert args.rope_scaling_factor is not None, "You have rope scaling type set but you didn't set a rope scaling factor, please use any floating point number > 1."
        assert args.rope_scaling_type in rope_scaling_valid_types, "Your rope scaling type setting is not valid, please use one of " + str(rope_scaling_valid_types) + "."
        assert isinstance(args.rope_scaling_factor, float) or isinstance(args.rope_scaling_factor, int), "Your rope scaling factor setting is not a number."
        assert args.rope_scaling_factor > 1, "Your rope scaling factor is less than or equal to 1, please use a higher setting."

    separator = '=' * 80
    accelerator.print('\nModel args:')
    accelerator.print(separator)
    accelerator.print(model_args)
    accelerator.print(separator)

    accelerator.print('\nData args:')
    accelerator.print(separator)
    accelerator.print(data_args)
    accelerator.print(separator)

    accelerator.print('\nTraining args:')
    accelerator.print(separator)
    accelerator.print(training_args)
    accelerator.print(separator)

    if accelerator.is_local_main_process:
        if accelerator.is_main_process:
            print(f"Main process here with index {accelerator.process_index}.")
        else:
            print(f"Local main process here with index {accelerator.process_index}.")
    else:
        print(f"Worker process here with index {accelerator.process_index}.")

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir, accelerator)
    if completed_training:
        accelerator.print('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir, accelerator)

    model.config.use_cache = False
    accelerator.print('Loaded model.')
    set_seed(args.seed)

    data_module = make_data_module(tokenizer, args, accelerator)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
    )
    accelerator = trainer.accelerator
    if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
        accelerator.print(f">>>>> DeepSpeed training with ZeRO stage {accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']['stage']}... <<<<<")

    # Callback
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback(trainer))

    # Verifying the datatypes and parameter counts before training.
    try:
        print_trainable_parameters(args, model, accelerator)
        dtypes = {}
        for _, p in model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes: dtypes[dtype] = 0
            dtypes[dtype] += p.numel()
        total = 0
        for k, v in dtypes.items(): total+= v
        for k, v in dtypes.items():
            accelerator.print(k, v, v/total)
    except Exception:
        pass

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Full fine-tune saving
    if args.full_finetune:
        accelerator.wait_for_everyone()
        accelerator.print("Saving full fine-tuned model...")
        if getattr(trainer, "deepspeed"):
            accelerator.print('>>>>> DeepSpeed saving... <<<<<')
            state_dict = accelerator.get_state_dict(trainer.deepspeed)
            unwrapped_model = accelerator.unwrap_model(trainer.deepspeed)
        else:
            state_dict = trainer.accelerator.get_state_dict(trainer.model)
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        if accelerator.is_main_process:
            save_dir = os.path.join(args.output_dir, "final")
            unwrapped_model.save_pretrained(save_dir, state_dict=state_dict, safe_serialization=args.save_as_safetensors, max_shard_size=args.max_shard_size)
            tokenizer.save_pretrained(save_dir)
        accelerator.wait_for_everyone()
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'], metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        with open(os.path.join(args.output_dir, "predictions.jsonl"), "w", encoding="utf8") as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if accelerator.is_local_main_process and (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf8") as fout:
            fout.write(json.dumps(all_metrics))

def main():
    try:
        train()
    except KeyboardInterrupt:
        print("Interrupted by user with sigint, script terminated.")

if __name__ == "__main__":
    main()
