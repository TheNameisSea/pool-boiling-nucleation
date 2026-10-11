#!/usr/bin/env bash
# Run a script from the repository root with upstream Bubbleformer on PYTHONPATH.
#   bin/run_python.sh scripts/rollout_diagnostics.py --checkpoint ... --job 735
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/external/Bubbleformer:${PYTHONPATH:-}"
cd "$ROOT"
python "$@"
