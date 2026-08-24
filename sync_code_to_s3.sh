#!/bin/bash
# Mirror this repo to the S3 code path that koala jobs pull from
# (s3://arcwm-code-us-west-2/axiom/code/Minecraft-CoT/).
#
# NOTE: the W&B key is hardcoded in trl_sft/common.sh / trl_sft/launch.sh and
# syncs to S3 with the scripts (intentional). Local secret files (.env.wandb,
# .env, .envrc, *.env.local) are still excluded to avoid accidental leakage.
#
# Usage:
#   bash sync_code_to_s3.sh            # safe full-repo differential sync
#   DRY=1 bash sync_code_to_s3.sh      # preview what would change
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_S3_URI="${CODE_S3_URI:-s3://arcwm-code-us-west-2/axiom/code}"
DEST="${CODE_S3_URI%/}/Minecraft-CoT/"

cd "$REPO_ROOT"

EXCLUDES=(
  --exclude '.git/**'
  --exclude '**/.env.wandb'
  --exclude '**/.env'
  --exclude '**/.envrc'
  --exclude '**/*.env.local'
  --exclude '**/__pycache__/**'
  --exclude '**/*.pyc'
  --exclude '**/.cache/**'
  --exclude '**/output/**'
  --exclude '**/.pytest_cache/**'
  --exclude '**/.ruff_cache/**'
  --exclude '**/.mypy_cache/**'
)

if [ "${DRY:-0}" = "1" ]; then
  echo "[sync] DRY-RUN -> $DEST"
  s5cmd --dry-run sync "${EXCLUDES[@]}" ./ "$DEST"
else
  echo "[sync] -> $DEST"
  s5cmd sync "${EXCLUDES[@]}" ./ "$DEST"
fi
echo "[sync] done. Verify no .env files leaked:"
s5cmd ls "${DEST}trl_sft/" 2>/dev/null | grep -iE "\.env" || echo "  (no .env files in trl_sft/ — good)"
