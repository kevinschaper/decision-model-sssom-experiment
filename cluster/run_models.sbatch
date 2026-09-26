#!/bin/bash
# LongLeaf: serve each model on the GPU node and run the item sets against it.
#   MODELS and ITEMS env vars (space-separated) select what runs; JevK5 skips the wide sets (<=16 options).
#   Kev runs its CUDA path here (flash-linear-attention); the Mac MLX run of kev-4b on `items` is the parity check.
#SBATCH --job-name=kevmap-run
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --time=08:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --output=/work/users/k/s/kschaper/logs/kevmap-run-%j.log
set -uo pipefail
# Self-contained env: LongLeaf home is 50 GB, so every cache lives on /work scratch.
[ -f ~/translator-env.sh ] && source ~/translator-env.sh
export WORK="/work/users/k/s/kschaper"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/.uv-cache}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORK/.uv-python}"
export HF_HOME="$WORK/.hf"
PROJ="$WORK/kev-mapping-experiment"; cd "$PROJ"
PORT=8009
MODELS=${MODELS:-"kev-0.8b kev-4b kev-9b hopper jevk5 jevk5-2b"}
ITEMS=${ITEMS:-"items items-v2 wide64 wide250"}
echo "### run on $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv

serve() {
  case "$1" in
    hopper)   hopper/.venv/bin/hopper-serve --adapter HopitAI/hopper --port $PORT ;;
    jevk5)    jevk5/.venv/bin/jevk5-serve --model alibiserikbay/JevK5 --port $PORT ;;
    jevk5-2b) jevk5/.venv/bin/jevk5-serve --model alibiserikbay/JevK5-2B --port $PORT ;;
    kev-*)    (cd kev && uv run --extra serve python -m kev.serve --run jaredpalmer/$1 --port $PORT) ;;
  esac
}
for m in $MODELS; do
  echo "### serve $m $(date)"
  serve "$m" > "results/serve-$m.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do curl -s -m 2 localhost:$PORT/v1/models >/dev/null && break; sleep 5; done
  curl -s localhost:$PORT/v1/models | head -c 400; echo
  for s in $ITEMS; do
    case "$m:$s" in jevk5*:wide*|hopper:wide*) echo "skip $m $s (option-letter limit)"; continue;; esac
    echo "### run $m $s $(date)"
    split=0; [ "$m" = hopper ] && split=1   # Hopper's server takes exactly one question per request
    KEVMAP_SPLIT_QUESTIONS=$split uv run kevmap run --model "$m" --items "$s" --port $PORT --workers 1 2>&1 | tail -3
  done
  kill $SPID 2>/dev/null; sleep 5; pkill -f -- "--port $PORT" 2>/dev/null; sleep 5
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
echo "### all done $(date)"
