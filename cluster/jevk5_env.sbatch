#!/bin/bash
# Rebuild the JevK5 venv on torch 2.8 (cu12.8) so the causal_conv1d kernel wheel fits; jevk5[fast] alone pulls a
# newer torch with no matching wheel, leaving the slow reference conv path (~0.9 s/item on an L40S).
#SBATCH --job-name=jevk5-env
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/users/k/s/kschaper/logs/jevk5-env-%j.log
set -uo pipefail
[ -f ~/translator-env.sh ] && source ~/translator-env.sh
export WORK="/work/users/k/s/kschaper"; export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/.uv-cache}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORK/.uv-python}"
cd "$WORK/kev-mapping-experiment/jevk5"
rm -rf .venv && uv venv --python 3.12 -q .venv
uv pip install -p .venv/bin/python torch==2.8.0 "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.2.0" \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  && .venv/bin/python -c "import torch, causal_conv1d; print('torch', torch.__version__, 'causal_conv1d', causal_conv1d.__version__)" \
  && .venv/bin/jevk5-serve --help | head -2
