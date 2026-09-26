#!/usr/bin/env bash
# Wait for the 0.8B run, swap the server to 4B, run 4B. Watchdog in `kevmap run` aborts if the server exceeds KEVMAP_MAX_SERVER_GB.
set -uo pipefail
cd "$(dirname "$0")/.."
until ! pgrep -f "kevmap run --model kev-0.8b" >/dev/null; do sleep 10; done
echo "0.8b run finished $(date)"
pkill -9 -f "serve_kev.py --run jaredpalmer/kev-0.8b"; sleep 5
(cd kev && KEV_DTYPE=bf16 uv run --extra serve python ../scripts/serve_kev.py --run jaredpalmer/kev-4b --port 8009 > ../results/serve-4b.log 2>&1) &
for i in $(seq 1 120); do curl -s -m 2 localhost:8009/v1/models >/dev/null && break; sleep 5; done
echo "4b server up $(date)"; curl -s localhost:8009/v1/models | python3 -c "import sys,json; m=json.load(sys.stdin)['models'][0]; print(m['run'], m['base'], m['backend'], m['dtype'])"
KEVMAP_MAX_SERVER_GB=16 uv run kevmap run --model kev-4b --workers 1 > results/run-4b.log 2>&1
echo "4b run exit $? $(date)"; tail -1 results/run-4b.log
