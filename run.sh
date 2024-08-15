export MODEL_PATH='Qwen/Qwen1.5-0.5B-Chat'
export SAVE_PATH='./output'
# export MASTER_ADDR="localhost"
export MASTER_PORT="1233"
# export GLOO_SOCKET_IFNAME="lo"
# export NCCL_SOCKET_IFNAME="lo"
# export CUDA_VISIBLE_DEVICES=1 
# export WANDB_DISABLED=true
# wandb offline

    # python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=1 --use_env train_math.py \
    # torchrun --nproc_per_node=1 --master_port ${MASTER_PORT} train_math.py \
    deepspeed --include localhost:1 --master_port ${MASTER_PORT} train_math.py --deepspeed ds_config.json\
    --model_name_or_path $MODEL_PATH \
    --data_path "./data/train/train.json" \
    --data_length 512 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to none \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.02 \
    --warmup_ratio 0.03 \
    --tf32 True \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

# python eval_gsm8k.py --model $SAVE_PATH --data_path ./data/test/GSM8K_test.jsonl
# python eval_math.py --model $SAVE_PATH --data_path ./data/test/MATH_test.jsonl


# train_micro_batch_size_per_gpu = args.per_device_train_batch_size
# train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps