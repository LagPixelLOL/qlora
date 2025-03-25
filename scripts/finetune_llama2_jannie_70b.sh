NCCL_P2P_DISABLE=1 accelerate launch --num_processes 8 --num_cpu_threads_per_process 30 qlora.py \
    --ddp_find_unused_parameters False \
    --model_name_or_path TheBloke/Llama-2-70B-fp16 \
    --output_dir ./output/LLaMA-2-Jannie-70B-QLoRA \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 722 \
    --save_total_limit 10 \
    --eval_strategy steps \
    --eval_dataset_size 1024 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
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
    --lr_scheduler_type constant \
    --adam8bit \
    --gradient_checkpointing \
    --dataset /home/ubuntu/jannie-log/jannie_log_train.jsonl \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps 3610 \
    --eval_steps 361 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0