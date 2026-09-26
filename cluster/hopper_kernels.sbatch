#!/bin/bash
# Hopper refuses to serve without the causal_conv1d CUDA kernel (compute capability 8.9 on the L40S). Install the
# prebuilt wheel that matches the venv (cu12 / torch 2.8 / cp312 / cxx11abi) and verify the import.
#SBATCH --job-name=hopper-kernels
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --mem=8g
#SBATCH --cpus-per-task=2
#SBATCH --output=/work/users/k/s/kschaper/logs/hopper-kernels-%j.log
set -uo pipefail
[ -f ~/translator-env.sh ] && source ~/translator-env.sh
export WORK="/work/users/k/s/kschaper"; export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/.uv-cache}"
cd "$WORK/kev-mapping-experiment/hopper"
uv pip install -p .venv/bin/python "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  && .venv/bin/python -c "import causal_conv1d; print('causal_conv1d ok', causal_conv1d.__version__)"
