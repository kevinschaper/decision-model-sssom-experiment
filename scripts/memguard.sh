#!/usr/bin/env bash
# Kill the kev run and server if the server's footprint (top MEM) exceeds $1 GB. Logs a reading every minute.
limit=${1:-16}
while pgrep -f "kevmap run" >/dev/null; do
  p=$(pgrep -f "serve_kev.py|kev.serve" | tail -1)
  if [ -n "$p" ]; then
    v=$(top -l 1 -pid "$p" -stats mem | tail -1 | tr -d ' +-')
    case "$v" in *G) gb=${v%G};; *M) gb=$(python3 -c "print(${v%M}/1024)");; *) gb=0;; esac
    echo "$(date +%H:%M:%S) server ${gb} GB swap $(sysctl -n vm.swapusage | awk '{print $6}')"
    if python3 -c "import sys; sys.exit(0 if float('$gb') > $limit else 1)"; then
      echo "ABORT: server ${gb} GB > ${limit} GB; killing run and server"
      pkill -f "kevmap run"; pkill -9 -f "serve_kev.py|kev.serve"
    fi
  fi
  sleep 60
done
echo "memguard done $(date)"
