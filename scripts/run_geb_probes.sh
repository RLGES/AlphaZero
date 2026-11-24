#!/usr/bin/env bash
# After the test-suite build: e-graph method comparison (egglog, all-rules-capped,
# MCTS-GEB, untrained AZ-GEB) on every suite. A fixed random sample of 100 T3 terms at 1,000 and 300
# e-nodes, then 50 of T4/T7 and all 29 of T5 at 1,000. Saves after every term (resumable). Results checkpoints/geb_probe_*.json/.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. RLGES_ASM_LEARNED=benchmarks/asm/learned_rules_qwen_all.json
# egglog default and BackOff-200 here; the trained AZ-GEB runs separately (TAG=_azv7).
# The re-implemented MCTS-GEB was dropped (24 Nov): the head-to-head uses MCTS-GEB's original code.
export ONLY=egglog,capped
W="${WORKERS:-6}"
for job in "T3 all 1000 100" "T3 all 300 100" "T4 all 1000 50" "T5 all 1000 29" "T7 all 1000 50"; do
  read -r suite which limit n <<< "$job"
  echo "=== $(date '+%F %T') probe $suite $which $limit"
  nice -n 10 .venv/bin/python scripts/geb_t3_probe.py "$n" "$limit" "$W" "$suite" "$which" 2>&1 \
    | grep -v "memory alloc\|RUST_BACK" | tee "checkpoints/geb_probe_${suite}_${which}_${limit}.log"
done
echo "=== $(date '+%F %T') probes done"
