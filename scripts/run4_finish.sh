#!/usr/bin/env bash
# Skip llmft and llmbig (decided 23 Nov): when run4.sh reaches the llmft step, stop
# it (only that PID and its descendants) and run the final evaluation for the
# builtin and llm arms. Replaces run4_skip_llmbig.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN4_PID="$1"
descendants() { local p; for p in $(pgrep -P "$1"); do echo "$p"; descendants "$p"; done; }
until grep -q "asm_v4_llmft" checkpoints/run4.log 2> /dev/null; do
  kill -0 "$RUN4_PID" 2> /dev/null || break
  sleep 30
done
pids="$RUN4_PID $(descendants "$RUN4_PID")"
kill $pids 2> /dev/null
sleep 10
kill -9 $pids 2> /dev/null
echo "=== $(date '+%F %T') llmft and llmbig skipped by decision: stopped run4.sh and children ($pids)" | tee -a checkpoints/run4.log
rm -rf checkpoints/asm_v4_llmft_s0
echo "=== $(date '+%F %T') final evaluation (builtin, llm)" | tee -a checkpoints/run4.log
TAG=v4 ARMS="builtin=none llm=benchmarks/asm/learned_rules_qwen_all.json" scripts/final_eval.sh 2>&1 | tee -a checkpoints/run4.log
.venv/bin/python scripts/ablation_table.py builtin='checkpoints/asm_v4_builtin_s*' \
  llm='checkpoints/asm_v4_llm_s*' > checkpoints/ablation_v4.md 2> /dev/null || true
echo "=== $(date '+%F %T') run 4 done" | tee -a checkpoints/run4.log
