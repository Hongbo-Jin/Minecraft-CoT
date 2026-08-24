# Minecraft-CoT

**Multi-stage supervised fine-tuning (SFT) pipeline for Minecraft vision-language / chain-of-thought agents.**

This repository trains Minecraft vision-language models (VLMs) that reason over game
frames and emit structured actions (and, in the full-trajectory stage, multi-step
chain-of-thought). It is built on top of the
[OpenHA](https://github.com/CraftJarvis/OpenHA) infrastructure: the Minecraft
environment, agent rollout, and grounding (SAM2) support come from the vendored
`openagents/`, `CrossAgent/`, and `external/` directories, while our primary
contribution is the `trl_sft/` training pipeline described below.

> The original OpenHA documentation is preserved at [`README_OpenHA.md`](./README_OpenHA.md)
> for reference (install steps, vLLM inference, output modes).

---

## Repository layout

| Path | Description | Origin |
|------|-------------|--------|
| `trl_sft/` | **Our SFT training pipeline** (Stage I / II / III) for Minecraft text-action VLMs. | Ours |
| `openagents/` | Minecraft agent framework, vLLM rollout client, system prompts. | Vendored (OpenHA) |
| `examples/` | Rollout / evaluation scripts (e.g. `rollout_openha.py`). | Vendored (OpenHA) |
| `scripts/` | Launch helpers for OpenHA inference. | Vendored (OpenHA) |
| `CrossAgent/` | Cross-level reinforcement-learning training code. | Vendored (OpenHA) |
| `external/` | External dependencies (modified SAM2 for grounding). | Vendored (OpenHA) |

---

## Installation

First install the OpenHA base, which provides `minestudio`, `openagents`, `vllm`, etc.:

```sh
git clone --recurse-submodules https://github.com/AxiomJin/Minecraft-CoT.git
conda create -n minecraft-cot python=3.10
conda activate minecraft-cot
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda install --channel=conda-forge openjdk=8 -y
pip install -e .
```

Then install the SFT training dependencies:

```sh
cd trl_sft
pip install -r requirements.txt
# optional, faster attention backend (compiled wheel needs packaging + ninja + psutil):
pip install flash-attn==2.7.4.post1
```

---

## SFT Training (`trl_sft/`)

`trl_sft/train_sft.py` is a [TRL](https://github.com/huggingface/trl) `SFTTrainer`-based
trainer for vision-language models. It supports two data layouts:

- **jsonl flat QA** (e.g. `minecraft-vlp`): prompt/completion form; loss is applied only
  to the final `"Action: ..."` assistant turn.
- **parquet trajectories** (e.g. `minecraft-text-action-dataset`): full-trajectory
  multi-step loss via `--full_trajectory`, training on *every* assistant turn across the
  whole episode (this is where chain-of-thought / multi-step reasoning is learned).

### Three-stage curriculum

| Stage | Data | Key flags | DeepSpeed | Notes |
|-------|------|-----------|-----------|-------|
| **Stage I** | text-only QA (`mc-qa-*.jsonl`) | `--text_only --freeze_vision_tower` | `ds_zero1.json` | World-knowledge post-training; ViT + adapter frozen |
| **Stage II** | jsonl VLM (`mc-vqa`, `mc-caption`, `mc-grounding-*`) | `--data_format jsonl --max_turns 4` | `ds_zero2.json` | Vision-language SFT on image + text |
| **Stage III** | parquet full trajectories (`minecraft-text-action-dataset`) | `--data_format parquet --full_trajectory` | `ds_zero2_no_offload.json` | Multi-step loss over whole trajectories |

### Supported backbones & data

- **Models**: Qwen2-VL, Qwen2.5-VL, Qwen3-VL, and Qwen3.5 VLMs (e.g. `Qwen2-VL-7B-Instruct`,
  `Qwen3.5-9B`). Weights and datasets live on S3 under `s3://arcwm-code-us-west-2/axiom/...`.
- **Datasets**: `minecraft-vlp` (jsonl: `mc-qa`, `mc-vqa`, `mc-caption`, `mc-grounding-*`)
  and `minecraft-text-action-dataset` (parquet full-trajectory episodes).

### Running

Each stage has a ready-to-run launcher that handles environment bootstrap, S3 data
localization (to local NVMe for throughput), and the DeepSpeed / `torchrun` launch:

```sh
# Stage I — text-only, freezes the vision tower
bash train_stage1.sh --nproc 8

# Stage II — jsonl vision-language SFT
bash train_stage2.sh --nproc 8

# Stage III — full-trajectory parquet (requires a Stage II checkpoint)
MODEL_PATH=s3://.../stage2-qwen35-9b  bash train_stage3.sh --nproc 8

# Use the Flash-Attention 2 backend:
bash train_stage2.sh --nproc 8 --attn-impl flash_attention_2
```

**Common overrides** (environment variables): `MODEL_PATH`, `DATA_PATH`, `OUTPUT_DIR`,
`MAX_STEPS`, `PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `MAX_SEQ_LENGTH`,
`DATALOADER_NUM_WORKERS`. `train_stage2.sh` derives `MAX_STEPS` from the dataset size and
the effective batch size automatically.

**W&B**: metrics are logged automatically — `WANDB_API_KEY` is hardcoded in
`trl_sft/common.sh` (sourced by all stage scripts), so both local and remote koala jobs
pick it up without extra setup. Override the dashboard run name with `WANDB_RUN_NAME`
(each stage script also sets a sensible default).

The single-file `trl_sft/launch.sh` contains the same three stages behind a `MODE`
argument for reference.

### Remote (koala) submission

The scripts target multi-GPU nodes (DeepSpeed ZeRO). Submit via koala, exporting
`WANDB_API_KEY` and using the relevant `train_stage*.sh` as the entry command.

---

## Inference / Rollout

After training, serve the model with vLLM and run rollouts / evaluation with
`examples/rollout_openha.py`. See [`README_OpenHA.md`](./README_OpenHA.md) for the original
OpenHA inference instructions and the available `--output_mode` / `system_message_tag`
schemes (`text_action`, `grounding_action`, `motion_action`, `grounding_coa`, `motion_coa`).

---

## Acknowledgements

This codebase builds on [OpenHA](https://github.com/CraftJarvis/OpenHA) and its
[MineStudio](https://github.com/CraftJarvis/MineStudio) environment, plus
[ROCKET-1](https://github.com/CraftJarvis/ROCKET-1) and
[SAM2](https://github.com/facebookresearch/sam2). We thank the CraftJarvis team for
open-sourcing the base framework that this work extends.

## Citation

If you use Minecraft-CoT in your work, please cite:

```bibtex
@misc{minecraft-cot,
  title        = {Minecraft-CoT: Multi-Stage SFT for Minecraft Vision-Language Agents},
  author       = {Jin, Hongbo},
  howpublished = {\url{https://github.com/AxiomJin/Minecraft-CoT}},
  year         = {2025}
}
```
