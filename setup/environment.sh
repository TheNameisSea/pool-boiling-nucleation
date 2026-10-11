#!/usr/bin/env bash
# Create the conda environment used for all runs (python 3.10, torch 2.5.0 + CUDA 12.4).
set -euo pipefail
ENV="${1:-nucleation}"
conda create -y -n "$ENV" python=3.10
conda run -n "$ENV" python -m pip install --upgrade pip
conda run -n "$ENV" python -m pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu124
conda run -n "$ENV" python -m pip install -r "$(dirname "$0")/../requirements.txt"
conda run -n "$ENV" python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
