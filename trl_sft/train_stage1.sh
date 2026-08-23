#!/bin/bash
# Stage I: text-only SFT, freeze vision tower.
# Usage:
#   bash train_stage1.sh [--nproc N] [--nnodes N] [--node-rank R] [--master-addr A] [--master-port P] [--attn-impl sdpa|flash_attention_2]
#
# W&B: if WANDB_API_KEY is set (via trl_sft/.env.wandb for local runs, or exported in the
# koala submit command for remote runs), metrics are logged automatically. Override the
# dashboard run name with WANDB_RUN_NAME; a per-stage default is applied below.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B-stage1-8gpu-20260817}"
DATA_PATH="${DATA_PATH:-s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-qa-241102.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp}"
OUTPUT_DIR="${OUTPUT_DIR:-./stage1-output}"
MAX_STEPS="${MAX_STEPS:-1000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3584}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export WANDB_RUN_NAME="${WANDB_RUN_NAME:-minecraft-sft-stage1-qwen35-9b}"

echo "=== Stage I training: NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC=$NPROC ==="
bootstrap_env
echo "MODEL_PATH=$MODEL_PATH OUTPUT_DIR=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS"
torchrun \
    --nnodes="$NNODES" --nproc_per_node="$NPROC" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    train_sft.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --data_format jsonl --text_only --freeze_vision_tower \
        --image_root "$IMAGE_ROOT" \
        --download_model "$DOWNLOAD_CACHE" \
        --output_dir "$OUTPUT_DIR" \
        --resume_from_checkpoint auto \
        --attn_implementation "$ATTN_IMPL" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --gradient_checkpointing \
        --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
        --num_train_epochs 1 \
        --max_steps "$MAX_STEPS" \
        --learning_rate 8e-6 \
        --deepspeed ds_zero1.json \
        --save_steps 999 \
        --logging_steps 10
