#!/usr/bin/env bash
# AZ -> egglog and AZ-chosen rules -> egglog on the full held-out and stress sets for
# llm s0-s2 (waits for each run's report_after.txt). Low priority, 3 workers.
# AZ chooses AZRULES_K=10 learned rules (built-in rules always kept), picked on 60
# held-out terms: 87.1 vs 86.9 for egglog with all rules, with ~35% fewer matches.
set -euo pipefail
cd "$(dirname "$0")/.."
export RLGES_ASM_LEARNED=benchmarks/asm/learned_rules_qwen_all.json PYTHONPATH=. AZRULES_K=10
for seed in 0 1 2; do
  run=checkpoints/asm_v4_llm_s$seed
  until [[ -s $run/report_after.txt ]]; do sleep 300; done
  echo "=== $(date '+%F %T') AZ-first hybrids for $run"
  nice -n 15 .venv/bin/python -c "
from pathlib import Path
from benchmarks.asm.final_eval import evaluate_az_first
evaluate_az_first([Path('$run')], ['heldout', 'stress'], (10, 25, 50, 100), workers=3)
"
done
echo "=== $(date '+%F %T') AZ-first hybrids done"
