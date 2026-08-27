#!/bin/bash
# Stage III: full-trajectory multi-step loss on parquet.
# Usage:
#   MODEL_PATH=<stage2 checkpoint> bash train_stage3.sh [--nproc N] [--attn-impl sdpa|flash_attention_2]
#
# W&B: see train_stage1.sh header. A per-stage default run name is applied below.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
DATA_PATH="${DATA_PATH:-s3://arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset/data/train-*.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-./stage3-output}"
MAX_STEPS="${MAX_STEPS:-3400}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-19456}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

: "${MODEL_PATH:?stage3 needs an explicit MODEL_PATH (a Stage II checkpoint), e.g. MODEL_PATH=s3://.../stage2-qwen35-9b bash train_stage3.sh}"

export WANDB_RUN_NAME="${WANDB_RUN_NAME:-minecraft-sft-stage3-qwen35-9b}"

# Upload every saved checkpoint to S3 right after it's written (crash-safe resume:
# a 48h deadline kill loses at most the in-flight checkpoint). Override with
# S3_OUTPUT_DIR=... or disable with S3_OUTPUT_DIR="" (empty string).
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-s3://arcwm-code-us-west-2/axiom/model/$WANDB_RUN_NAME}"

echo "=== Stage III training (full-trajectory multi-step loss): NPROC=$NPROC ==="
bootstrap_env
localize_stage3_parquet_dataset
echo "MODEL_PATH=$MODEL_PATH OUTPUT_DIR=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS S3_OUTPUT_DIR=$S3_OUTPUT_DIR"

# Crash-safe resume: if the local output dir has no checkpoint yet but S3 does
# (e.g. a previous run was killed by the 48h deadline), pull the latest checkpoints
# down first so `--resume_from_checkpoint auto` can pick them up.
if [ -n "$S3_OUTPUT_DIR" ] && ! compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null 2>&1; then
    echo "No local checkpoint; pulling latest from S3: $S3_OUTPUT_DIR"
    LATEST_CKPT=$(aws s3 ls "$S3_OUTPUT_DIR/" 2>/dev/null | grep -oE 'checkpoint-[0-9]+' | sort -t- -k2 -n | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Latest: $LATEST_CKPT"
        if command -v s5cmd >/dev/null 2>&1; then
            s5cmd sync --concurrency 16 "${S3_OUTPUT_DIR%/}/${LATEST_CKPT}/*" "$OUTPUT_DIR/${LATEST_CKPT}/"
        else
            aws s3 sync "$S3_OUTPUT_DIR/$LATEST_CKPT" "$OUTPUT_DIR/$LATEST_CKPT" --only-show-errors
        fi
        if [ -f "$OUTPUT_DIR/$LATEST_CKPT/latest" ]; then
            echo "Checkpoint downloaded OK: $LATEST_CKPT"
        else
            echo "ERROR: checkpoint download failed" >&2
            exit 1
        fi
    else
        echo "No checkpoint found on S3, starting from scratch."
    fi
fi

torchrun --nproc_per_node="$NPROC" --tee 3 train_sft.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --data_format parquet \
    --download_model "$DOWNLOAD_CACHE" \
    --output_dir "$OUTPUT_DIR" \
    --s3_output_dir "$S3_OUTPUT_DIR" \
    --resume_from_checkpoint auto \
    --full_trajectory \
    --attn_implementation "$ATTN_IMPL" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --learning_rate 8e-6 \
    --weight_decay 0.05 \
    --warmup_steps 102 \
    --lr_scheduler_type cosine \
    --deepspeed ds_zero2_no_offload.json \
    --save_steps 200 \
    --logging_steps 10
