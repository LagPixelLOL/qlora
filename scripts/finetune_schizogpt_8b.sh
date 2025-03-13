PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes 8 --num_cpu_threads_per_process 30 --use_deepspeed --gradient_accumulation_steps 2 --zero_stage 2 --zero3_save_16bit_model True qlora.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --output_dir ./output/SchizoGPT-8B-QLoRA \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 58 \
    --save_total_limit 10 \
    --eval_strategy steps \
    --eval_dataset_size 512 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 512 \
    --dataloader_num_workers 30 \
    --group_by_length \
    --logging_strategy steps \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --adam8bit \
    --gradient_checkpointing \
    --dataset /home/ubuntu/r-chatgpt-general-dump/merged_strings_train.jsonl \
    --source_max_len 1 \
    --target_max_len 131071 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 576 \
    --eval_steps 58 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --additional_special_tokens '<username>' '<message>'