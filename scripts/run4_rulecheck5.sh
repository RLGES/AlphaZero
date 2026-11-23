#!/usr/bin/env bash
# After discovery round 5: merge its rules and score rule sets by plain MCTS on
# 300 train-split terms (scripts/rule_select_eval.py). Writes checkpoints/rulecheck5.log.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=benchmarks/asm
until [[ -f "$D/learned_rules_qwen5.json" ]]; do sleep 60; done
"$PY" scripts/merge_rules.py --out "$D/learned_rules_qwen_all5.json" \
  "$D/learned_rules_qwen.json" "$D/learned_rules_qwen2.json" "$D/learned_rules_qwen3.json" \
  "$D/learned_rules_qwen4.json" "$D/learned_rules_qwen5.json"
"$PY" scripts/pool_rules.py "$D/learned_rules_pool5.json" \
  "$D"/learned_rules_qwen.json "$D"/learned_rules_qwen{2,3,4,5}.json
echo "=== $(date '+%F %T') rule check (MCTS, 300 train-split terms, seed 424242)"
for r in "$D/learned_rules_qwen_all.json" "$D/learned_rules_qwen_all5.json" "$D/learned_rules_pool5.json"; do
  PYTHONPATH=. RLGES_ASM_LEARNED="$r" nice -n 10 "$PY" scripts/rule_select_eval.py 300 3
done
echo "=== $(date '+%F %T') rule check done"
