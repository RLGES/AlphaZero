#!/usr/bin/env bash
# After the test-suite build: e-graph method comparison (egglog, all-rules-capped,
# MCTS-GEB, untrained AZ-GEB) on every suite. All of T3 at 1,000 and 300 e-nodes,
# then up to 150 terms of T4, T5, T7 at 1,000. Results checkpoints/geb_probe_*.json/.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. RLGES_ASM_LEARNED=benchmarks/asm/learned_rules_qwen_all.json
until grep -q "test suites done" checkpoints/testsets.log; do sleep 120; done
W="${WORKERS:-6}"
for job in "T3 all 1000 400" "T3 all 300 400" "T4 all 1000 150" "T5 all 1000 150" "T7 all 1000 150"; do
  read -r suite which limit n <<< "$job"
  echo "=== $(date '+%F %T') probe $suite $which $limit"
  nice -n 10 .venv/bin/python scripts/geb_t3_probe.py "$n" "$limit" "$W" "$suite" "$which" 2>&1 \
    | grep -v "memory alloc\|RUST_BACK" | tee "checkpoints/geb_probe_${suite}_${which}_${limit}.log"
done
echo "=== $(date '+%F %T') probes done"
