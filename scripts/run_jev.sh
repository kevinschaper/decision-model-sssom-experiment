#!/usr/bin/env bash
# Score Jev (TypeSafe's hosted System-One model) on the item sets. The bearer key comes from JEV_API_KEY in the
# shell env (sourced from ~/.zshrc); it is never written anywhere. Hard stop at KEVMAP_MAX_INPUT_TOKENS.
set -uo pipefail
cd "$(dirname "$0")/.."
# ~/.zshrc is zsh; read the variable through a zsh subshell rather than sourcing it in bash
JEV_API_KEY=$(zsh -c 'source ~/.zshrc >/dev/null 2>&1; printf %s "$JEV_API_KEY"')
[ -n "${JEV_API_KEY:-}" ] || { echo "JEV_API_KEY not set"; exit 1; }
export KEVMAP_API_KEY="$JEV_API_KEY" KEVMAP_MAX_INPUT_TOKENS=${KEVMAP_MAX_INPUT_TOKENS:-60000000}
for s in ${SETS:-items-v2 items k8lab wide64 wide250}; do
  echo "$(date '+%F %T') run jev $s"
  uv run kevmap run --model jev --items "$s" --base-url https://api.typesafe.ai --request-model jev-latest --workers 2 2>&1 | tail -2
done
echo "$(date '+%F %T') jev done"
