#!/usr/bin/env bash
# Evaluate one trained run as in the paper (one GPU, ~10 min per mode).
#
#   bin/evaluate_run.sh <run id> <checkpoint (epoch=149-step=150000.ckpt)>
#
# Writes results/diagnostics/model_diagnostics_<run>_minarea16{,_teacherforced}.csv and saves the
# predicted fields to $WORK_ROOT/rollout_fields_<run> and $WORK_ROOT/tf_fields_<run>, which the
# analysis scripts read. Runs 735 and 739 also get the per-step reinitialisation rollout (Fig. 4);
# runs 735, 716, 739 and 778 also get the 4-cell-threshold rollout (Section 3.3).
set -euo pipefail
RUN="$1"; CKPT="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="${WORK_ROOT:-$ROOT/work}"
D="${BUBBLEML_ROOT:-$ROOT/data/bubbleml2}"
PY="$ROOT/bin/run_python.sh"
COMMON=(--checkpoint "$CKPT" --job "$RUN" --label "run$RUN")
if [ "$RUN" = 778 ]; then
  COMMON+=(--data-dir "$D/SingleBubble-Saturated-FC72-2D" --sims 90 92 94 96 98 100
           --reference results/diagnostics/singlebubble_reference_diagnostics.csv)
fi
if [ "$RUN" != 974 ]; then      # 974 is trained on the first output frame only: no rollout
  "$PY" scripts/rollout_diagnostics.py "${COMMON[@]}" --save-fields "$W/rollout_fields_$RUN"
fi
"$PY" scripts/rollout_diagnostics.py "${COMMON[@]}" --teacher-forced --save-fields "$W/tf_fields_$RUN"
case "$RUN" in 735|739) "$PY" scripts/rollout_diagnostics.py "${COMMON[@]}" --project ;; esac
case "$RUN" in 735|716|739|778) "$PY" scripts/rollout_diagnostics.py "${COMMON[@]}" --min-area 4 ;; esac
