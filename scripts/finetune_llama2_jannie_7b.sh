NCCL_P2P_DISABLE=1 accelerate launch --num_processes 8 --num_cpu_threads_per_process 30 qlora.py \
    --ddp_find_unused_parameters False \
    --model_name_or_path TheBloke/Llama-2-7B-fp16 \
    --output_dir ./output/llama-2-jannie-7b \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 10 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 30 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval False \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset /home/ubuntu/jannie-log/jannie_log_train.jsonl \
    --source_max_len 1024 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0
