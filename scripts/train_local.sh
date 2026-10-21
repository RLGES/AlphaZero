#!/usr/bin/env bash
# Train AlphaZero suites one after another on this machine, resuming any
# suite that already has a checkpoint. Logs: checkpoints/<suite>_v1/train.log
#
#   nohup scripts/train_local.sh arithmetic boolean bitvector > checkpoints/queue.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS="${ROUNDS:-20}"
WORKERS="${WORKERS:-7}"

for suite in "$@"; do
  out="checkpoints/${suite}_v1"
  mkdir -p "$out"
  resume=()
  [[ -f "$out/latest.pt" ]] && resume=(--resume)
  echo "$(date "+%F %T") start $suite ${resume[@]+"${resume[*]}"}"
  .venv/bin/python -m src.engines.alphazero.train --suite "$suite" --rounds "$ROUNDS" \
    --games 256 --sims 64 --steps 400 --workers "$WORKERS" --out "$out" \
    ${resume[@]+"${resume[@]}"} \
    >> "$out/train.log" 2>&1
  echo "$(date '+%F %T') done $suite"
done
