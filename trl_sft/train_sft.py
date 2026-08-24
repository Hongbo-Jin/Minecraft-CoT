"""
TRL-based SFT training for Minecraft text-action VLM.

Data (two supported layouts, see `build_minecraft_dataset`):
  - parquet trajectories: s3://arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset/
  - jsonl flat QA (e.g. minecraft-vlp): s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/
Model: s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B/ (also works for Qwen2-VL /
    Qwen2.5-VL / Qwen3-VL under s3://arcwm-code-us-west-2/axiom/model/)

See `launch.sh` for ready-to-run `debug`/`train` (Stage I/II)/`stage3` (Stage III,
full-trajectory multi-step loss) invocations, including env bootstrap + S3 localization.

Usage (Stage II, prompt/completion, only the final "Action: ..." turn trained on):
    torchrun --nproc_per_node=$NPROC train_sft.py \
        --model_path s3://.../Qwen3.5-9B --output_dir ./output \
        --data_path s3://.../minecraft-text-action-dataset/data/train-*.parquet \
        --max_turns 4 --max_seq_length 16384 \
        --per_device_batch_size 2 --gradient_accumulation_steps 4 \
        --num_train_epochs 1 --deepspeed ds_zero2.json

    # jsonl / flat-QA layout (e.g. minecraft-vlp); --data_format auto-detects from the
    # ".jsonl" extension, --image_root defaults to the dir containing --data_path:
    torchrun --nproc_per_node=$NPROC train_sft.py \
        --model_path s3://.../Qwen2-VL-7B-Instruct --output_dir ./output \
        --data_path s3://.../minecraft-vlp/mc-vqa-241102.jsonl

    # Stage I (JARVIS-VLA "world knowledge" text-only post-training): plain text QA,
    # no images (e.g. mc-qa-*.jsonl). --text_only omits the "images" key entirely (so
    # SFTTrainer uses its plain-text collator) and --freeze_vision_tower keeps ViT+
    # adapter frozen. Hyperparameters below match the paper (LR=5e-6, beta2=0.95/wd=0,
    # 200-step warmup, global batch=256, ZeRO-1) except grad_accum is scaled up 4x to
    # compensate for 8 GPUs here vs. the paper's 32 (same effective global batch, just
    # slower wall-clock); ds_zero1.json hardcodes its WarmupDecayLR schedule to match
    # --warmup_steps/--max_steps below (a plain HF scheduler under DeepSpeed was found
    # to blow through warmup 5-16x faster than requested on this stack -- update the
    # json if you change those args).
    torchrun --nproc_per_node=8 train_sft.py \
        --model_path s3://.../Qwen3.5-9B --output_dir ./output \
        --data_path s3://.../minecraft-vlp/mc-qa-250312.jsonl \
        --text_only --freeze_vision_tower \
        --max_seq_length 3584 --per_device_batch_size 2 --gradient_accumulation_steps 16 \
        --num_train_epochs 1 --max_steps 1077 --learning_rate 5e-6 --lr_scheduler_type cosine \
        --warmup_steps 200 --weight_decay 0.0 --adam_beta2 0.95 --max_grad_norm 1.0 \
        --seed 42 --deepspeed ds_zero1.json

    # Stage III (full-trajectory multi-step loss): see --full_trajectory below.

NOTE on VLM + TRL:
  - All target models are vision-language models, so TRL's `SFTTrainer` always picks
    `DataCollatorForVisionLanguageModeling` (triggered by a top-level "images" key).
    That collator does not support `packing=True` or `assistant_only_loss=True` for
    VLMs (both raise `ValueError` at trainer-init time) -- so this script uses one of
    two loss-masking strategies instead:
      * default: split each sample into {"prompt": [...context...], "completion":
        [last assistant turn]} + `SFTConfig(completion_only_loss=True)`, which TRL DOES
        support for VLMs (`_collate_prompt_completion`) -- loss only on the target
        "Action: ..." turn.
      * --full_trajectory: keep the WHOLE trajectory as one flat `messages` list and
        let `MultiStepVLMCollator` (subclasses `DataCollatorForVisionLanguageModeling`)
        mask every non-assistant token to -100 itself, training on EVERY assistant turn.
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from collators import (
    ImmutableVisionCollatorAdapter,
    MultiStepVLMCollator,
    freeze_vision_tower,
)
from dataset import build_minecraft_dataset
from diagnostics import HeartbeatCallback, install_stall_watchdog

logger = logging.getLogger(__name__)


def _local_cache_name(s3_or_local_path: str) -> str:
    """Derive a filesystem-safe, model-specific cache dir name from a path."""
    name = s3_or_local_path.rstrip("/").split("/")[-1]
    return name or "model"


def download_from_s3(s3_path: str, local_dir: str, exclude_checkpoints: bool = False) -> str:
    """Download model/dataset from S3 to local disk, skip if already exists.

    `exclude_checkpoints=True` skips any `checkpoint-*/` subdirectory -- relevant when
    `s3_path` is a PREVIOUS run's `--output_dir` used as the next stage's starting
    model: it contains both the final merged model at the root (everything
    `from_pretrained` reads) AND every intermediate `--save_steps` checkpoint (a full
    DeepSpeed ZeRO checkpoint with fp32 optimizer state, ~5x the plain model size each,
    up to `--save_total_limit` of them). Downloading those is pure waste (verified: one
    real 9B-model output dir was 475GB total, only 18.8GB of it actually needed).
    """
    local_dir = Path(local_dir)
    marker = local_dir / ".download_complete"

    if marker.exists():
        logger.info(f"Already downloaded: {local_dir}")
        return str(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {s3_path} -> {local_dir} ...")
    exclude_flag = " --exclude 'checkpoint-*/*'" if exclude_checkpoints else ""
    ret = os.system(f"aws s3 cp --recursive{exclude_flag} {s3_path} {local_dir}/")
    if ret != 0:
        raise RuntimeError(f"aws s3 cp failed (exit={ret}) for {s3_path} -> {local_dir}")
    marker.touch()
    logger.info(f"Download complete: {local_dir}")
    return str(local_dir)


def _resolve_max_len_kwarg(max_seq_length: int) -> Dict[str, int]:
    """Detect whether SFTConfig uses `max_length` or `max_seq_length` (renamed in newer TRL)."""
    sft_config_field_names = {f.name for f in dataclass_fields(SFTConfig)}
    if "max_length" in sft_config_field_names:
        return {"max_length": max_seq_length}
    if "max_seq_length" in sft_config_field_names:
        return {"max_seq_length": max_seq_length}
    logger.warning("Neither `max_length` nor `max_seq_length` found on SFTConfig; skipping.")
    return {}


def _load_model_and_processor(args):
    """Download (if S3), load model + processor, optionally freeze vision tower."""
    local_model_path = args.model_path
    if args.model_path.startswith("s3://"):
        cache_dir = args.download_model or f"/tmp/{_local_cache_name(args.model_path)}"
        local_model_path = download_from_s3(args.model_path.rstrip("/"), cache_dir, exclude_checkpoints=True)
    logger.info(f"Loading model from {local_model_path} ...")
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        local_model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    if args.freeze_vision_tower:
        freeze_vision_tower(model)
    return model, processor


def _build_dataset(args, processor):
    """Build the Minecraft SFT dataset (streaming, unsharded)."""
    return build_minecraft_dataset(
        data_path=args.data_path,
        max_turns=args.max_turns,
        streaming=True,
        seed=args.seed,
        data_format=args.data_format,
        image_root=args.image_root,
        text_only=args.text_only,
        processor=processor,
        max_seq_length=args.max_seq_length,
        full_trajectory=args.full_trajectory,
    )


def _setup_collator(trainer, args, processor):
    """Set the appropriate data collator on the trainer."""
    if args.full_trajectory:
        trainer.data_collator = MultiStepVLMCollator(processor=processor, max_length=args.max_seq_length)
    elif not args.text_only:
        trainer.data_collator = ImmutableVisionCollatorAdapter(trainer.data_collator)


# ─── main training ────────────────────────────────────────────────────────────


def _is_complete_trainer_checkpoint(checkpoint: str) -> bool:
    """Return whether a Trainer checkpoint has its completion state and model artifact."""
    trainer_state_path = os.path.join(checkpoint, "trainer_state.json")
    if not os.path.isfile(trainer_state_path) or os.path.getsize(trainer_state_path) == 0:
        return False
    try:
        with open(trainer_state_path, "r", encoding="utf-8") as state_file:
            json.load(state_file)
    except (OSError, json.JSONDecodeError):
        return False

    model_artifacts = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        # DeepSpeed ZeRO checkpoint marker: written by engine.save_checkpoint only
        # after all shards are flushed, so its presence means the checkpoint is
        # complete for DeepSpeed runs (which have no model.safetensors/pytorch_model.bin).
        "latest",
    )
    return any(os.path.isfile(os.path.join(checkpoint, artifact)) for artifact in model_artifacts)


def _resolve_resume_checkpoint(requested: Optional[str], output_dir: str) -> Optional[str]:
    """Resolve an explicit checkpoint path or the latest complete checkpoint in ``output_dir``."""
    if requested in (None, "none"):
        return None

    if requested == "auto":
        candidates = []
        if os.path.isdir(output_dir):
            for entry in os.scandir(output_dir):
                if entry.is_dir() and entry.name.startswith("checkpoint-"):
                    try:
                        step = int(entry.name.removeprefix("checkpoint-"))
                    except ValueError:
                        continue
                    candidates.append((step, entry.path))
        for _, checkpoint in sorted(candidates, reverse=True):
            if _is_complete_trainer_checkpoint(checkpoint):
                logger.info(f"Resuming exact Trainer state from checkpoint: {checkpoint}")
                return checkpoint
            logger.warning(f"Skipping incomplete checkpoint: {checkpoint}")
        logger.warning("--resume_from_checkpoint=auto found no complete checkpoint; starting a new run.")
        return None

    if not os.path.isdir(requested):
        raise ValueError(f"Resume checkpoint does not exist or is not a directory: {requested}")
    if not _is_complete_trainer_checkpoint(requested):
        raise ValueError(f"Resume checkpoint is incomplete: {requested}")
    logger.info(f"Resuming exact Trainer state from checkpoint: {requested}")
    return requested


class S3CheckpointUploadCallback(TrainerCallback):
    """Upload each saved checkpoint to S3 right after it's written (crash-safe resume).

    `on_save` fires after the Trainer writes each `checkpoint-*`; we kick off an
    async `aws s3 sync` so the upload never blocks training. A 48h deadline kill then
    loses at most the in-flight checkpoint, and `--resume_from_checkpoint auto` can
    pick up the latest complete one from S3 on the next run.
    """

    def __init__(self, s3_output_dir: str):
        self.s3_output_dir = s3_output_dir.rstrip("/")
        self._last_proc: Optional[subprocess.Popen] = None
        # Prefer s5cmd (much faster concurrent upload) when available; fall back to aws cli.
        self._use_s5cmd = shutil.which("s5cmd") is not None

    def _sync_command(self, output_dir: str):
        if self._use_s5cmd:
            # s5cmd sync needs the source dir to end with "/" to sync its contents.
            return [
                "s5cmd", "sync", "--concurrency", "16",
                output_dir.rstrip("/") + "/", self.s3_output_dir + "/",
            ]
        return ["aws", "s3", "sync", output_dir, self.s3_output_dir, "--only-show-errors"]

    def on_save(self, args, state, control, **kwargs):
        output_dir = args.output_dir
        if not os.path.isdir(output_dir):
            return
        # Avoid overlapping syncs: wait for the previous one if it's still running.
        if self._last_proc is not None and self._last_proc.poll() is None:
            logger.info("Waiting for previous S3 upload to finish before starting the next...")
            self._last_proc.wait()
        self._last_proc = subprocess.Popen(
            self._sync_command(output_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Started async S3 upload: {output_dir} -> {self.s3_output_dir}")

    def finalize(self, output_dir: str):
        """Wait for the last upload, then do a final full sync (final model + processor)."""
        if self._last_proc is not None:
            self._last_proc.wait()
        ret = subprocess.call(
            self._sync_command(output_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ret != 0:
            logger.warning(f"Final S3 upload failed (exit={ret}) for {output_dir} -> {self.s3_output_dir}")
        else:
            logger.info(f"Final S3 upload complete: {output_dir} -> {self.s3_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="TRL SFT for Minecraft VLM")
    parser.add_argument("--model_path", type=str, required=True, help="S3 or local path to model")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="S3 glob or local path to parquet/jsonl files. May be a comma-separated "
        "list of several such paths/globs (e.g. to combine minecraft-vlp's "
        "mc-vqa-*.jsonl + mc-caption-*.jsonl + mc-grounding-point-*.jsonl into one "
        "dataset for JARVIS-VLA Stage II) -- each entry is passed through as-is (still "
        "supports globs), just don't mix parquet and jsonl in the same list.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="auto",
        choices=["auto", "parquet", "jsonl"],
        help="'parquet' (e.g. minecraft-text-action-dataset, embedded image_bytes) or "
        "'jsonl' (e.g. minecraft-vlp, images loaded from an 'image_root' via relative "
        "paths). 'auto' (default) infers from --data_path's extension.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Only used for --data_format=jsonl: base dir/URI that each row's 'image' "
        "relative path(s) are resolved against. Defaults to the directory containing "
        "--data_path (matches minecraft-vlp's <root>/*.jsonl + <root>/images/ layout).",
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Stage I mode (JARVIS-VLA's 'Minecraft world knowledge' text-only "
        "post-training): treat every row as plain system+user+assistant text QA with "
        "no images (e.g. minecraft-vlp/mc-qa-*.jsonl), producing samples with no "
        "'images' key so SFTTrainer uses its plain-text collator instead of the "
        "vision-language one. Typically combined with --freeze_vision_tower.",
    )
    parser.add_argument(
        "--freeze_vision_tower",
        action="store_true",
        help="Freeze the vision encoder + adapter/merger, training only the LLM "
        "backbone -- matches JARVIS-VLA's Stage I recipe. Typically combined with "
        "--text_only.",
    )
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="auto",
        help="Checkpoint directory to resume exactly (model, optimizer, scheduler and RNG). "
        "'auto' (default) resumes the latest complete checkpoint-* in --output_dir; "
        "use 'none' to force a new run.",
    )
    parser.add_argument("--max_turns", type=int, default=4, help="Max (user,assistant) pairs per sample")
    parser.add_argument(
        "--full_trajectory",
        action="store_true",
        help="Stage III multi-step loss: keep the WHOLE parquet trajectory (no random "
        "history window) and train on EVERY assistant 'Action: ...' turn via "
        "MultiStepVLMCollator, instead of only the last turn. Requires --max_seq_length "
        "large enough to hold a full trajectory (e.g. 19456).",
    )
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Trade compute for activation memory -- recommended for large models / "
        "long max_seq_length, especially when DeepSpeed optimizer-state offload isn't "
        "available (e.g. due to a CUDA-toolkit/torch-build version mismatch preventing "
        "DeepSpeedCPUAdam's JIT compile).",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Explicit total training steps, overriding the built-in dataset-size "
        "estimate below (which is specific to minecraft-text-action-dataset and wrong "
        "for any other dataset -- always pass this for --text_only/--data_format=jsonl runs).",
    )
    parser.add_argument("--learning_rate", type=float, default=8e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Fixed warmup step count. If > 0, this takes precedence over --warmup_ratio "
        "(standard `transformers.TrainingArguments` behavior). JARVIS-VLA's Stage I/II "
        "recipe uses a fixed 200-step warmup rather than a ratio.",
    )
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="HF Trainer default is 0.999; JARVIS-VLA's recipe uses 0.95.",
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Background DataLoader worker processes (per rank) for prefetch/"
        "preprocessing (image decode/resize, tokenization) while the GPU trains on the "
        "previous batch. Bump this (e.g. 8) if GPU utilization is low after confirming "
        "images are read from local disk, not S3.",
    )
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stall_dump_seconds",
        type=float,
        default=180.0,
        help="Diagnose hangs: if no training step completes within this many seconds, "
        "dump every thread's Python traceback to stderr/job log (NCCL's own watchdog "
        "only reports a timeout, never which line each rank is stuck on). Keep below "
        "the NCCL timeout (600s). Set 0 to disable.",
    )
    parser.add_argument("--download_model", type=str, default=None, help="Local dir to cache downloaded model")
    parser.add_argument(
        "--s3_output_dir",
        type=str,
        default=None,
        help="If set, upload each saved checkpoint to this S3 prefix right after it's "
        "written (async, non-blocking), and do a final full sync after training. "
        "Enables crash-safe resume: a 48h deadline kill loses at most the in-flight "
        "checkpoint, and --resume_from_checkpoint auto can pick up the latest complete "
        "one from S3 on the next run.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="'sdpa' (default) needs no extra install and avoids a confirmed flash-attn "
        "wheel/torch ABI mismatch on the koala training image (undefined symbol at "
        "import time). Pass 'flash_attention_2' only after verifying a matching build.",
    )
    # NOTE: TRL's SFTTrainer raises ValueError for packing=True on vision-language
    # models (all supported models here are VLMs), so packing defaults to False and
    # any attempt to force it on is rejected with a clear error instead of a crash
    # deep inside the trainer.
    parser.add_argument("--packing", action="store_true", default=False, help="NOT supported for VLM training; kept for API symmetry")
    parser.add_argument("--no_packing", action="store_false", dest="packing")

    args = parser.parse_args()

    # Arm before anything heavy, so even a hang during dataset/model setup is caught.
    install_stall_watchdog(args.stall_dump_seconds)

    # Allow --data_path to be a comma-separated list of files/globs (e.g. combining
    # VQA + Caption + Grounding jsonls for JARVIS-VLA Stage II, which don't share a
    # single glob pattern without also pulling in unrelated files like mc-qa-*.jsonl).
    # `build_minecraft_dataset`/`_detect_data_format`/`_default_image_root` all accept
    # either a plain str or a List[str] here.
    if "," in args.data_path:
        args.data_path = [p.strip() for p in args.data_path.split(",") if p.strip()]

    if args.packing:
        raise ValueError(
            "--packing was requested, but TRL's SFTTrainer does not support sequence "
            "packing for vision-language models (Qwen2-VL / Qwen2.5-VL / Qwen3-VL / "
            "Qwen3.5-VL are all VLMs here). Remove --packing."
        )

    # ── seed ──
    set_seed(args.seed)

    resume_from_checkpoint = _resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)

    # ── model + dataset ──
    model, processor = _load_model_and_processor(args)
    dataset = _build_dataset(args, processor)

    # ── training config ──
    total_batch_size = args.per_device_batch_size * args.gradient_accumulation_steps
    # For torchrun, world_size is available via env
    n_gpus = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", 1)))
    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        # Compute max_steps from approximate dataset size. This magic number is
        # `minecraft-text-action-dataset`-specific (363 files x ~600 samples each ~=
        # 217800 samples per epoch) -- it does NOT apply to other datasets/formats
        # (e.g. --text_only Stage I data, which has a very different row count). Pass
        # --max_steps explicitly for anything other than the default parquet dataset.
        if args.text_only or args.data_format == "jsonl":
            logger.warning(
                "No --max_steps given and --text_only/--data_format=jsonl is set: "
                "falling back to the minecraft-text-action-dataset-specific sample-count "
                "estimate below, which is almost certainly wrong for this dataset. Pass "
                "--max_steps explicitly."
            )
        approx_dataset_size = 217800
        max_steps = (approx_dataset_size * args.num_train_epochs) // (total_batch_size * n_gpus)

    max_len_kwarg = _resolve_max_len_kwarg(args.max_seq_length)

    # `TrainingArguments.warmup_ratio` was REMOVED as an `__init__` kwarg in some
    # transformers releases (consistent with a "warmup_ratio is deprecated ... removed
    # in v5.2" warning seen on affected versions) in favor of `warmup_steps` alone --
    # passing `warmup_ratio=...` unconditionally would then raise `TypeError` at
    # `SFTConfig(...)` construction time. Detect support the same way `max_length` vs
    # `max_seq_length` is detected above, and only pass whichever of
    # `warmup_steps`/`warmup_ratio` is actually needed: if `--warmup_steps > 0` is
    # requested, always pass that (unambiguous on every version tested); otherwise fall
    # back to `--warmup_ratio`, but only include it in the SFTConfig kwargs if the
    # installed version still supports it (recent versions default warmup_ratio's
    # effect to 0 when omitted, which is the same as passing 0.0 explicitly anyway).
    warmup_kwargs: Dict[str, float] = {"warmup_steps": args.warmup_steps}
    if args.warmup_steps <= 0:
        if "warmup_ratio" in {f.name for f in dataclass_fields(SFTConfig)}:
            warmup_kwargs["warmup_ratio"] = args.warmup_ratio
        else:
            logger.warning(
                "--warmup_ratio was requested but this installed version of "
                "`trl`/`transformers` no longer accepts `warmup_ratio` on `SFTConfig`; "
                "ignoring it. Pass --warmup_steps (an absolute step count) instead."
            )
    logger.info(f"Warmup config: {warmup_kwargs}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        deepspeed=args.deepspeed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        packing=False,  # unsupported for VLMs, see argparse note above
        # prompt/completion path masks the context out of the loss (only the target
        # "Action: ..." completion is trained on). --full_trajectory instead builds the
        # labels inside `MultiStepVLMCollator`, so TRL's own masking must be OFF.
        completion_only_loss=not args.full_trajectory,
        seed=args.seed,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
        run_name=os.environ.get("WANDB_RUN_NAME", "minecraft-sft-trl"),
        **warmup_kwargs,
        **max_len_kwarg,
    )

    logger.info(f"Training config: total_batch={total_batch_size}, n_gpus={n_gpus}, max_steps={max_steps}")
    logger.info(
        f"Resolved training_args: warmup_steps={training_args.warmup_steps}, "
        f"warmup_ratio={getattr(training_args, 'warmup_ratio', 'N/A')}, "
        f"max_steps={training_args.max_steps}, learning_rate={training_args.learning_rate}"
    )

    # ── trainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    _setup_collator(trainer, args, processor)
    if args.stall_dump_seconds > 0:
        trainer.add_callback(HeartbeatCallback())
    s3_upload_callback = None
    if args.s3_output_dir:
        s3_upload_callback = S3CheckpointUploadCallback(args.s3_output_dir)
        trainer.add_callback(s3_upload_callback)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    if s3_upload_callback is not None:
        s3_upload_callback.finalize(args.output_dir)

    logger.info(f"Training finished. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
