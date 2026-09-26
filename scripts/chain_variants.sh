#!/usr/bin/env bash
# After the current 4B `items` run: 4B on items-v2 / wide64 / wide250, then swap to 0.8B for the same three.
# Each run gets its own memguard. Item sets must exist (built by `kevmap build --name ...`).
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "$(date +%H:%M:%S) $*"; }
wait_run() { until ! pgrep -f "kevmap run" >/dev/null; do sleep 20; done; }
serve() {  # size
  pkill -9 -f "serve_kev.py|kev.serve"; sleep 5
  (cd kev && KEV_DTYPE=bf16 uv run --extra serve python ../scripts/serve_kev.py --run jaredpalmer/kev-$1 --port 8009 > ../results/serve-$1.log 2>&1) &
  for i in $(seq 1 120); do curl -s -m 2 localhost:8009/v1/models >/dev/null && break; sleep 5; done
  log "server $1 up"
}
run() {  # model items
  log "run $1 $2"
  (scripts/memguard.sh 16 > results/memguard-$1-$2.log 2>&1 &)
  KEVMAP_MAX_SERVER_GB=16 uv run kevmap run --model $1 --items $2 --workers 1 > results/run-$1-$2.log 2>&1
  log "run $1 $2 exit $? $(tail -1 results/run-$1-$2.log)"
}
wait_run; log "items run finished"
for s in items-v2 wide64 wide250; do until [ -s data/$s.jsonl ]; do sleep 30; done; done
for s in items-v2 wide64 wide250; do run kev-4b $s; done
serve 0.8b
for s in items-v2 wide64 wide250; do run kev-0.8b $s; done
log "all variant runs done"
