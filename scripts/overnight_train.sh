#!/usr/bin/env bash
# Overnight on the Mac: fine-tune Kev-0.8B on the SNOMED->Mondo training records, then serve the checkpoint and
# score it on the eval sets. Usage: N_RECORDS=1500 EPOCHS=1 scripts/overnight_train.sh   (run under nohup)
set -uo pipefail
cd "$(dirname "$0")/.."
N=${N_RECORDS:-1500}; EPOCHS=${EPOCHS:-1}; LR=${LR:-2e-5}; NAME=${NAME:-kev-0.8b-snomed}
MAX_STATE=${MAX_STATE:-1536}; PORT=${KEV_PORT:-8009}
log() { echo "$(date '+%F %T') $*"; }
head -n "$N" data/train.jsonl > "data/train-$N.jsonl"
log "train $NAME on $N records, $EPOCHS epoch(s), lr $LR"
(cd kev && uv run --extra serve python -m kev.train --data "../data/train-$N.jsonl" --base Qwen/Qwen3.5-0.8B-Base \
   --init_from jaredpalmer/kev-0.8b --epochs "$EPOCHS" --lr "$LR" --batch 1 --accum 8 --device mps --weights_dtype bf16 \
   --checkpointing 1 --max_state "$MAX_STATE" --out "../runs/$NAME" > "../results/train-$NAME.log" 2>&1) &
TPID=$!
# memory guard for training: abort above 24 GB footprint (top MEM; the 0.8B run legitimately touched 21 GB), sampled each minute
while kill -0 $TPID 2>/dev/null; do
  v=$(top -l 1 -pid "$(pgrep -f "kev.train --data ../data/train-$N" | tail -1)" -stats mem 2>/dev/null | tail -1 | tr -d ' +-')
  case "$v" in *G) gb=${v%G};; *M) gb=$(python3 -c "print(${v%M}/1024)");; *) gb=0;; esac
  echo "$(date +%H:%M:%S) train ${gb} GB"
  if python3 -c "import sys; sys.exit(0 if float('$gb') > 24 else 1)"; then
    log "ABORT training: ${gb} GB"; pkill -9 -f "kev.train --data ../data/train-$N"; kill $TPID 2>/dev/null; break  # the python, not just the wrapper subshell
  fi
  sleep 60
done
wait $TPID; log "training exit $?"
[ -f "runs/$NAME/head.pt" ] || { log "no checkpoint at runs/$NAME; stopping"; exit 1; }
log "serve runs/$NAME"
(cd kev && KEV_DTYPE=bf16 uv run --extra serve python ../scripts/serve_kev.py --run "../runs/$NAME" --port "$PORT" > "../results/serve-$NAME.log" 2>&1) &
for i in $(seq 1 120); do curl -s -m 2 "localhost:$PORT/v1/models" >/dev/null && break; sleep 5; done
for s in items-v2 k8lab items; do
  log "run $NAME $s"
  (scripts/memguard.sh 16 > "results/memguard-$NAME-$s.log" 2>&1 &)
  KEVMAP_MAX_SERVER_GB=16 uv run kevmap run --model "$NAME" --items "$s" --port "$PORT" --workers 1 > "results/run-$NAME-$s.log" 2>&1
  log "run $NAME $s exit $? $(tail -1 "results/run-$NAME-$s.log")"
done
pkill -9 -f "serve_kev.py --run ../runs/$NAME"
for s in items-v2 k8lab items; do uv run kevmap eval --items "$s" > /dev/null 2>&1; done
log "overnight done"
