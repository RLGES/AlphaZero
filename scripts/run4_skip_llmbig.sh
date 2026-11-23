#!/usr/bin/env bash
# Stop run4.sh (only that PID and its children) when it reaches the llmbig step,
# after the main final evaluation has finished. llmbig is skipped by decision (23 Nov).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN4_PID="$1"
until grep -q "asm_v4_llmbig" checkpoints/run4.log 2> /dev/null; do
  kill -0 "$RUN4_PID" 2> /dev/null || { echo "run4.sh already exited"; exit 0; }
  sleep 30
done
kids=$(pgrep -P "$RUN4_PID")
grandkids=$(for k in $kids; do pgrep -P "$k"; done)
kill $RUN4_PID $kids $grandkids 2> /dev/null
sleep 5
for g in $grandkids; do pkill -P "$g" 2> /dev/null; done
echo "=== $(date '+%F %T') llmbig skipped: stopped run4.sh ($RUN4_PID) and its children"
echo "=== $(date '+%F %T') llmbig skipped by decision; run 4 baseline done" >> checkpoints/run4.log
