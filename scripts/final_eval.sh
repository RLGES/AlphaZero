#!/usr/bin/env bash
# Final night-3 evaluation: every finished asm_v3 run on all held-out terms and the
# stress set (benchmarks/asm/final_eval.py), then the tables.
#
#   scripts/final_eval.sh                 # WORKERS=7 by default; rerun to resume
#   WORKERS=1 nice -n 19 scripts/final_eval.sh   # alongside training
#
#   TAG=v4 ARMS="builtin=none llm=<rules>" scripts/final_eval.sh   # another run's arms
#
# Tables: checkpoints/final_<TAG>_heldout.md, checkpoints/final_<TAG>_stress.md
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-7}"
TAG="${TAG:-v3}"  # runs are checkpoints/asm_<TAG>_<arm>_s<seed>
if [[ -n "${ARMS:-}" ]]; then
  read -r -a arms <<< "$ARMS"  # "name=rules name=rules ..."
else
  arms=("builtin=none" "llm=benchmarks/asm/learned_rules_qwen.json"
        "llm2=benchmarks/asm/learned_rules_qwen12.json")
fi

specs=()
for arm in "${arms[@]}"; do
  name="${arm%%=*}" rules="${arm#*=}"
  pattern="checkpoints/asm_${TAG}_${name}_s*"
  specs+=("$name=$pattern")
  compgen -G "$pattern/report_after.txt" > /dev/null || continue
  echo "=== $(date '+%F %T') final eval: $name"
  RLGES_ASM_LEARNED="$rules" .venv/bin/python -m benchmarks.asm.final_eval \
    --runs "$pattern" --workers "$WORKERS" ${LIMIT:+--limit "$LIMIT"}
done

for set in heldout stress; do
  .venv/bin/python scripts/ablation_table.py --csv "final_${set}${LIMIT:+_limit$LIMIT}.csv" "${specs[@]}" \
    > "checkpoints/final_${TAG}_$set.md" 2>/dev/null || true
  echo "=== $set"; cat "checkpoints/final_${TAG}_$set.md"
done
