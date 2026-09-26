#!/bin/bash
# LongLeaf: build the envs for the cluster leg (CPU node; compute nodes have outbound internet).
#   kevmap runner env, Hopper, JevK5, Kev (CUDA path). Model weights are pre-downloaded to $WORK/.hf.
#SBATCH --job-name=kevmap-setup
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --output=/work/users/k/s/kschaper/logs/kevmap-setup-%j.log
set -uo pipefail
# Self-contained env: LongLeaf home is 50 GB, so every cache lives on /work scratch.
[ -f ~/translator-env.sh ] && source ~/translator-env.sh
export WORK="/work/users/k/s/kschaper"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/.uv-cache}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORK/.uv-python}"
export HF_HOME="$WORK/.hf"
PROJ="$WORK/kev-mapping-experiment"
cd "$PROJ"
echo "### setup on $(hostname) $(date)"

echo "### kevmap env"; uv sync --no-group dev && uv run kevmap --help | head -3

echo "### hopper"; mkdir -p hopper && cd hopper
[ -d src ] || git clone -q https://github.com/hopit-ai/hopper src
(cd src && git checkout -q v1.1.0)
uv venv --python 3.12 -q .venv
uv pip install -q -p .venv/bin/python torch==2.8.0 transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 flash-linear-attention==0.5.2 huggingface_hub
uv pip install -q -p .venv/bin/python --no-deps -e src
.venv/bin/hopper-serve --help | head -3
cd "$PROJ"

echo "### jevk5"; mkdir -p jevk5 && cd jevk5
uv venv --python 3.12 -q .venv
uv pip install -q -p .venv/bin/python "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.2.0"
.venv/bin/jevk5-serve --help | head -3
cd "$PROJ"

echo "### kev"; [ -d kev ] || git clone -q --depth 1 https://github.com/jaredpalmer/kev.git kev
cd kev && uv sync --extra serve -q && uv pip install -q flash-linear-attention==0.5.2 && cd "$PROJ"

echo "### weights"; hopper/.venv/bin/hf download HopitAI/hopper >/dev/null && hopper/.venv/bin/hf download Qwen/Qwen3.5-4B --include "*.json" "*.safetensors" "*.txt" "*.jinja" >/dev/null && echo hopper ok
hopper/.venv/bin/hf download alibiserikbay/JevK5 >/dev/null && hopper/.venv/bin/hf download alibiserikbay/JevK5-2B >/dev/null && echo jevk5 ok
hopper/.venv/bin/hf download jaredpalmer/kev-9b >/dev/null && hopper/.venv/bin/hf download Qwen/Qwen3.5-9B-Base --include "*.json" "*.safetensors" "*.txt" "*.jinja" >/dev/null && echo kev-9b ok
du -sh "$HF_HOME"
echo "### setup done $(date)"
