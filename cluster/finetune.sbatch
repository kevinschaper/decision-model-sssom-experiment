#!/bin/bash
# LongLeaf: fine-tune Kev from the released checkpoints on data/train.jsonl (eval-disjoint SNOMED->Mondo records),
# then serve each checkpoint on the node and score it. SIZES env var picks sizes (default "0.8b 4b").
#SBATCH --job-name=kevmap-finetune
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --time=08:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --output=/work/users/k/s/kschaper/logs/kevmap-finetune-%j.log
set -uo pipefail
[ -f ~/translator-env.sh ] && source ~/translator-env.sh
export WORK="/work/users/k/s/kschaper"; export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/.uv-cache}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORK/.uv-python}"
export HF_HOME="$WORK/.hf"
PROJ="$WORK/kev-mapping-experiment"; cd "$PROJ"; PORT=8009
SIZES=${SIZES:-"0.8b 4b"}; EPOCHS=${EPOCHS:-2}; LR=${LR:-2e-5}; SETS=${SETS:-"items-v2 k8lab items"}
echo "### finetune on $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for size in $SIZES; do
  NAME="kev-$size-snomed-full"; BASE="Qwen/Qwen3.5-${size^^}-Base"; BASE=${BASE/0.8B/0.8B}
  echo "### train $NAME from jaredpalmer/kev-$size ($BASE) $(date)"
  (cd kev && uv run --extra serve python -m kev.train --data ../data/train.jsonl --base "$BASE" --init_from "jaredpalmer/kev-$size" \
     --epochs "$EPOCHS" --lr "$LR" --batch 1 --accum 8 --dtype bf16 --checkpointing 1 --max_state 1536 --device cuda \
     --out "../runs/$NAME" 2>&1 | grep -aE "step [0-9]*0/|dropped|training requests|saved|Error|Traceback" | tail -40)
  [ -f "runs/$NAME/head.pt" ] || { echo "### no checkpoint for $NAME"; continue; }
  echo "### serve $NAME $(date)"
  (cd kev && uv run --extra serve python -m kev.serve --run "../runs/$NAME" --port $PORT > "../results/serve-$NAME.log" 2>&1) &
  SPID=$!
  for i in $(seq 1 120); do curl -s -m 2 localhost:$PORT/v1/models >/dev/null && break; sleep 5; done
  for s in $SETS; do
    echo "### run $NAME $s $(date)"
    uv run kevmap run --model "$NAME" --items "$s" --port $PORT --workers 1 2>&1 | tail -2
  done
  kill $SPID 2>/dev/null; sleep 5; pkill -f -- "--port $PORT" 2>/dev/null; sleep 5
done
echo "### all done $(date)"
