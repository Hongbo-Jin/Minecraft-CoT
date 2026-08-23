#!/bin/bash
# Shared bootstrap for train_stage1/2/3.sh. Do NOT run directly -- source it.
# Provides: arg parsing, W&B config, bootstrap_env, localize_* helpers.

set -euo pipefail

# ── defaults ──
MODE="${MODE:-train}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29400}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        stage1|train|stage3) MODE="$1"; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --attn-impl) ATTN_IMPL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/local-ssd/minecraft-vlp}"
LOCAL_PARQUET_ROOT="${LOCAL_PARQUET_ROOT:-/local-ssd/minecraft-text-action}"
DOWNLOAD_CACHE="${DOWNLOAD_CACHE:-/local-ssd/model_cache}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"

# W&B: sourced from a git-ignored local file (this repo is PUBLIC on GitHub -- never
# commit a real key). Create trl_sft/.env.wandb with WANDB_API_KEY/WANDB_PROJECT/
# WANDB_RUN_NAME. Only applies to LOCAL runs of this script; remote koala jobs must
# export WANDB_API_KEY explicitly in the submit command instead.
if [ -f "$SCRIPT_DIR/.env.wandb" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env.wandb"
fi

# Make the training env ready to run `torchrun` (idempotent -- skips whatever's already
# installed). The default koala training image ships NO torch/trl/deepspeed at all, and
# has no `openha` (or similar) conda env baked in -- both confirmed by trial and error,
# so this is the single source of truth going forward instead of re-discovering it via
# failed jobs. requirements.txt has flash-attn commented out (see its own comment) --
# --attn-impl sdpa (the default) does NOT need it installed at all.
bootstrap_env() {
    # The koala image runs under the C locale; TRL's create_model_card() (on every
    # --save_steps checkpoint) reads its model-card template with Path.read_text()'s
    # default encoding and crashes with UnicodeDecodeError under ascii. Export a UTF-8
    # locale so Python starts in UTF-8 mode (train_sft.py also sets it defensively).
    export LANG=C.UTF-8 LC_ALL=C.UTF-8
    if ! command -v conda >/dev/null 2>&1; then
        echo "[env] no conda found, assuming torch/trl/deepspeed are already on PATH." >&2
        return
    fi
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    conda env list | grep -q "^sft " || conda create -n sft python=3.10 -y -q
    conda activate sft
    python -c "import torch" 2>/dev/null || \
        pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    python -c "import trl, transformers, deepspeed, datasets, accelerate" 2>/dev/null || \
        pip install -q -r requirements.txt
}

# Pre-download every Stage II JSONL shard in `$DATA_PATH` plus every image
# `$IMAGE_ROOT` points to onto local SSD, then REPOINT `$DATA_PATH`/`$IMAGE_ROOT` at the
# local copies. Must run on EACH node (local SSD is node-local). No-op if `$DATA_PATH`
# is already local.
localize_stage2_jsonl_and_images() {
    if [[ "$DATA_PATH" != s3://* ]]; then
        echo "[localize] DATA_PATH is already local, skipping." >&2
        return
    fi
    if ! python3 -c "import s3fs" 2>/dev/null; then
        echo "Missing s3fs: install trl_sft/requirements.txt before using S3 data or weights." >&2
        exit 1
    fi

    mkdir -p "$LOCAL_DATA_ROOT"
    echo "[localize] Downloading Stage II JSONL shard(s) to $LOCAL_DATA_ROOT ..." >&2
    local_paths=()
    IFS=',' read -ra shard_uris <<< "$DATA_PATH"
    for shard_uri in "${shard_uris[@]}"; do
        local_file="$LOCAL_DATA_ROOT/$(basename "$shard_uri")"
        if [ ! -s "$local_file" ]; then
            aws s3 cp "$shard_uri" "$local_file" --only-show-errors
        else
            echo "[localize] $local_file already present, skipping re-download." >&2
        fi
        local_paths+=("$local_file")
    done

    echo "[localize] Syncing referenced images from $IMAGE_ROOT to $LOCAL_DATA_ROOT ..." >&2
    if command -v s5cmd >/dev/null 2>&1; then
        s5cmd sync --exclude "*.jsonl" "${IMAGE_ROOT%/}/*" "$LOCAL_DATA_ROOT/"
    else
        aws s3 sync "${IMAGE_ROOT%/}/" "$LOCAL_DATA_ROOT/" --exclude "*.jsonl" --only-show-errors
    fi

    DATA_PATH="$(IFS=,; echo "${local_paths[*]}")"
    IMAGE_ROOT="$LOCAL_DATA_ROOT"
    echo "[localize] Done. DATA_PATH=$DATA_PATH IMAGE_ROOT=$IMAGE_ROOT" >&2
}

# Pre-download `minecraft-text-action-dataset`'s parquet shards (Stage III) onto local
# SSD, then repoint `$DATA_PATH` at the local glob -- same rationale as
# `localize_stage2_jsonl_and_images` above. Each parquet row embeds its own
# `image_bytes` column, so no separate image sync is needed. Expects `$DATA_PATH` to be
# a SINGLE glob (e.g. "s3://.../data/train-*.parquet"), not a comma-separated list.
localize_stage3_parquet_dataset() {
    if [[ "$DATA_PATH" != s3://* ]]; then
        echo "[localize] DATA_PATH is already local, skipping." >&2
        return
    fi

    local s3_dir="${DATA_PATH%/*}/"
    local glob_name="$(basename "$DATA_PATH")"
    mkdir -p "$LOCAL_PARQUET_ROOT"
    echo "[localize] Syncing parquet shards from $s3_dir to $LOCAL_PARQUET_ROOT ..." >&2
    if command -v s5cmd >/dev/null 2>&1; then
        s5cmd sync --include "*.parquet" "${s3_dir}*" "$LOCAL_PARQUET_ROOT/"
    else
        aws s3 sync "$s3_dir" "$LOCAL_PARQUET_ROOT/" --exclude "*" --include "*.parquet" --only-show-errors
    fi

    # Quoted below at the call site -- an UNQUOTED glob here would be expanded by bash
    # into hundreds of positional args before argparse ever sees it (confirmed root
    # cause of a real launch failure: `--data_path` is a single str that does its own
    # internal glob matching, not something the shell should expand for it).
    DATA_PATH="$LOCAL_PARQUET_ROOT/$glob_name"
    echo "[localize] Done. DATA_PATH=$DATA_PATH" >&2
}
