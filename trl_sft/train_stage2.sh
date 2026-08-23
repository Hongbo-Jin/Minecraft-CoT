#!/bin/bash
# Stage II: VLM SFT on jsonl VQA/caption/grounding.
# Usage:
#   bash train_stage2.sh [--nproc N] [--nnodes N] [--node-rank R] [--master-addr A] [--master-port P] [--attn-impl sdpa|flash_attention_2]
#
# W&B: see train_stage1.sh header. A per-stage default run name is applied below.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B-stage1-8gpu-20260817}"
DATA_PATH="${DATA_PATH:-s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-vqa-241102.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-caption-241104.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-grounding-point-embodied-image5.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-grounding-point-embodied.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-grounding-point-gui.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp}"
OUTPUT_DIR="${OUTPUT_DIR:-./stage2-qwen35-9b}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
STAGE2_TRAIN_SAMPLES="${STAGE2_TRAIN_SAMPLES:-261461}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# MAX_STEPS derived from sample count / effective batch size -- must be computed AFTER
# sourcing common.sh so NPROC/NNODES reflect parsed CLI overrides.
TOTAL_GPUS=$((NNODES * NPROC))
MAX_STEPS="${MAX_STEPS:-$((STAGE2_TRAIN_SAMPLES / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * TOTAL_GPUS)))}"

export WANDB_RUN_NAME="${WANDB_RUN_NAME:-minecraft-sft-stage2-qwen35-9b}"

echo "=== Stage II training: NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC=$NPROC ==="
bootstrap_env
localize_stage2_jsonl_and_images
echo "MODEL_PATH=$MODEL_PATH OUTPUT_DIR=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS TOTAL_GPUS=$TOTAL_GPUS"
torchrun \
    --nnodes="$NNODES" --nproc_per_node="$NPROC" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    train_sft.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --data_format jsonl \
        --image_root "$IMAGE_ROOT" \
        --download_model "$DOWNLOAD_CACHE" \
        --output_dir "$OUTPUT_DIR" \
        --resume_from_checkpoint auto \
        --attn_implementation "$ATTN_IMPL" \
        --max_turns 4 \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --gradient_checkpointing \
        --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
        --num_train_epochs 1 \
        --max_steps "$MAX_STEPS" \
        --learning_rate 8e-6 \
        --deepspeed ds_zero2.json \
        --save_steps 200 \
        --logging_steps 10
