#!/usr/bin/env bash
# Train one model with upstream Bubbleformer (150 epochs, batch size 2, one GPU), optionally with a
# modified training loss.
#
#   bin/train.sh model_cfg=<model> data_cfg=<data> [seed=<n>] [other Hydra overrides]
#   TRAIN_ENTRY=src/<wrapper>.py [loss env vars] bin/train.sh ...
#
# TRAIN_ENTRY selects the entry point (default: upstream scripts/train.py); the wrappers in src/
# change only the training criterion and read their settings from environment variables (see each
# file). configs/runs.csv lists the exact command for every run in the paper.
# Environment: BUBBLEML_ROOT (data), WORK_ROOT (checkpoints go to $WORK_ROOT/checkpoints).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_HOME="${BUBBLEML_ROOT:-$ROOT/data/bubbleml2}"
LOG_DIR="${WORK_ROOT:-$ROOT/work}/checkpoints"
UP="$ROOT/external/Bubbleformer"
[ -d "$UP" ] || { echo "run setup/get_bubbleformer.sh first" >&2; exit 1; }
export PYTHONPATH="$UP:${PYTHONPATH:-}"
ENTRY="${TRAIN_ENTRY:-scripts/train.py}"
case "$ENTRY" in src/*) ENTRY="$ROOT/$ENTRY" ;; esac
cd "$UP"
ARGS=("$ENTRY" expt=forecast nodes=1 devices=1 max_epochs="${MAX_EPOCHS:-150}" batch_size="${BATCH_SIZE:-2}"
      use_wandb=False +data_home="$DATA_HOME" log_dir="$LOG_DIR" "$@")
echo "upstream_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "loss_env: BAND_WEIGHT=${BAND_WEIGHT:-} BAND_ROWS=${BAND_ROWS:-} IND_WEIGHT=${IND_WEIGHT:-} IND_EPS=${IND_EPS:-} PF_EVERY=${PF_EVERY:-} TOL_SHIFT=${TOL_SHIFT:-} TOL_BAND_ROWS=${TOL_BAND_ROWS:-} TOL_TILE=${TOL_TILE:-}"
echo "INVOCATION: python ${ARGS[*]}"
python "${ARGS[@]}"
