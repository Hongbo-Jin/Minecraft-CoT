#!/bin/bash
# ============================================================================
# 评测 Stage II（VQA+grounding，尚未见过 text-action 轨迹数据）checkpoint 本身
# 的 TextVLA 动作输出能力 —— 作为 Stage III 训练前的基线对比。
#
# 模型信息：
#   来源: s3://.../Qwen2-VL-7B-Instruct-stage2-8gpu-20260817/
#         （本次修复双重分片bug后重跑得到的 Stage II checkpoint，
#          Stage III 训练的 MODEL_PATH 就是这份权重）
#   架构: Qwen2VLForConditionalGeneration (vllm 0.8.5 直接兼容)
#   预期: Stage II 没有在 text-action 轨迹数据上训练过，预计动作格式遵循度/
#         成功率会显著低于 Stage III 最终 checkpoint，用于量化 Stage III 的增益。
# ============================================================================
set -o pipefail
export MODEL_LOCAL_NAME="stage2-qwen2vl-7b-20260817"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/Qwen2-VL-7B-Instruct-stage2-8gpu-20260817/"
export SERVED_MODEL_NAME="eval-stage2-qwen2vl-7b"
export VLLM_CONDA_ENV="openha"   # Qwen2-VL 架构，vLLM==0.8.5 原生支持
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"
