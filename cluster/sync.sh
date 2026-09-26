#!/usr/bin/env bash
# Push code + item sets to LongLeaf scratch (git stays authoritative here); pull results back with --pull.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJ=/work/users/k/s/kschaper/kev-mapping-experiment
if [ "${1:-}" = "--pull" ]; then
  # cluster runs land as results/<model>@ll/<items>/ so they sit next to, never over, the Mac runs
  mkdir -p results/.ll
  rsync -az --include '*/' --include 'responses.jsonl' --include 'server.json' --exclude '*' longleaf:$PROJ/results/ results/.ll/
  for d in results/.ll/*/; do m=$(basename "$d"); mkdir -p "results/$m@ll"; rsync -a "$d" "results/$m@ll/"; done
  exit
fi
ssh longleaf "mkdir -p $PROJ/results $PROJ/data"
rsync -az --exclude '.git/' --exclude '.venv/' --exclude 'kev/' --exclude 'jevk5/' --exclude 'hopper/' \
  --exclude 'results/' --exclude 'data/' --exclude '__pycache__/' --exclude '.ruff_cache/' ./ longleaf:$PROJ/
rsync -az data/*.jsonl longleaf:$PROJ/data/
