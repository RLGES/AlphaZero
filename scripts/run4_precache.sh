#!/usr/bin/env bash
# Fill the final-eval caches (MCTS, egglog, egglog+MCTS on held-out and stress) for
# the builtin and qwen_all rule sets now, at low priority, while training runs.
# final_eval.sh later reuses them, and they give the rule lift on the night-3 test sets.
set -euo pipefail
cd "$(dirname "$0")/.."
for rules in none benchmarks/asm/learned_rules_qwen_all.json; do
  echo "=== $(date '+%F %T') caches for rules $rules"
  RLGES_ASM_LEARNED="$rules" PYTHONPATH=. nice -n 15 .venv/bin/python -c "
from benchmarks.asm.final_eval import evaluate_runs
evaluate_runs([], ['heldout', 'stress'], (10, 25, 50, 100), workers=3)
"
done
echo "=== $(date '+%F %T') caches done"
