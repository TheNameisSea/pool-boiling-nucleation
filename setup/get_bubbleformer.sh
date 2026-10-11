#!/usr/bin/env bash
# Clone upstream Bubbleformer at the commit used for every run in the paper, and copy this
# repository's data configs into its config tree. Upstream code is not modified otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UP="$ROOT/external/Bubbleformer"
COMMIT=8d1dade4c34b69f6048c6bcba04b62d63117efe5
if [ ! -d "$UP/.git" ]; then
  git clone https://github.com/HPCForge/Bubbleformer.git "$UP"
fi
git -C "$UP" checkout -q "$COMMIT"
cp -v "$ROOT"/configs/data_cfg/*.yaml "$UP/bubbleformer/config/data_cfg/"
echo "Bubbleformer at $(git -C "$UP" rev-parse --short HEAD) in $UP"
