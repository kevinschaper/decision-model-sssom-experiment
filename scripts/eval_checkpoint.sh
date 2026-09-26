#!/usr/bin/env bash
# Serve a local Kev checkpoint (runs/NAME) through the capped MLX wrapper and score it on the given item sets.
# Usage: scripts/eval_checkpoint.sh NAME [items-v2 k8lab items]   (run under nohup)
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1; shift; SETS=${*:-"items-v2 k8lab items"}; PORT=${KEV_PORT:-8009}
log() { echo "$(date '+%F %T') $*"; }
pkill -9 -f "serve_kev.py" 2>/dev/null; sleep 2
log "serve runs/$NAME"
(cd kev && KEV_DTYPE=bf16 uv run --extra serve python ../scripts/serve_kev.py --run "../runs/$NAME" --port "$PORT" > "../results/serve-$NAME.log" 2>&1) &
for i in $(seq 1 120); do curl -s -m 2 "localhost:$PORT/v1/models" >/dev/null && break; sleep 5; done
curl -s "localhost:$PORT/v1/models" | head -c 300; echo
for s in $SETS; do
  log "run $NAME $s"
  (scripts/memguard.sh 16 > "results/memguard-$NAME-$s.log" 2>&1 &)
  KEVMAP_MAX_SERVER_GB=16 uv run kevmap run --model "$NAME" --items "$s" --port "$PORT" --workers 1 > "results/run-$NAME-$s.log" 2>&1
  log "run $NAME $s exit $? $(tail -1 "results/run-$NAME-$s.log")"
done
pkill -9 -f "serve_kev.py --run ../runs/$NAME"
for s in $SETS; do uv run kevmap eval --items "$s" > /dev/null 2>&1; done
log "eval done"
