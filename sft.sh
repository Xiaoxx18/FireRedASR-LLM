# 4 * 35GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \

ENABLE_AUDIO_OUTPUT=true \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=29502 \
NPROC_PER_NODE=1 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
swift sft \
    --model  \
    --model_type fireredllm_asr \
    --template fireredllm_asr \
    --custom_register_path 'examples/fireredasr/fireredllm_asr.py' \
    --dataset    \
    --load_from_cache_file false \
    --dataset_shuffle true \
    --lazy_tokenize true \
    --split_dataset_ratio 0.01 \
    --loss_scale last_round \
    --train_type lora \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 200 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj' \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --gradient_accumulation_steps 1 \
    --eval_steps 1 \
    --save_steps 1 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir  \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --deepspeed zero3