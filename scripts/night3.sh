#!/usr/bin/env bash
# Night 3: the rule-set ablation repeated over training seeds, with a steadier best.pt.
#
#   scripts/night3.sh          # keep the laptop open and plugged in; rerun to resume
#   QUICK=1 scripts/night3.sh  # smoke run
#
# Arms (Learn-R hard + control-flow corpus):
#   builtin  Learn-R's 44 built-in rules (+ fold)
#   llm      + the 20 round-1 qwen rules            (benchmarks/asm/learned_rules_qwen.json)
#   llm2     + round-1 and round-2 qwen rules       (benchmarks/asm/learned_rules_qwen12.json, if present)
# builtin and llm go seed by seed (s0 of both, then s1, then s2), so stopping early
# still leaves complete comparisons; llm2 runs after them. Each run:
# checkpoints/asm_v3_<arm>_s<seed>.
# Versus night 2: best.pt scored on 300 held-out terms (was 100), patience 15 (was 12),
# reports on 150 generated terms per eval seed (was 50).
# Table at the end: checkpoints/ablation_v3.md
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2}"
export HELDOUT="${HELDOUT:-300}" PATIENCE="${PATIENCE:-15}" GENERATED="${GENERATED:-150}"
export EXTRA_SUITES=""

arms=("builtin=none" "llm=benchmarks/asm/learned_rules_qwen.json")
schedule=()
for seed in $SEEDS; do
  for arm in "${arms[@]}"; do schedule+=("$seed:$arm"); done
done
if [[ -f benchmarks/asm/learned_rules_qwen12.json ]]; then
  arms+=("llm2=benchmarks/asm/learned_rules_qwen12.json")
  for seed in $SEEDS; do schedule+=("$seed:llm2=benchmarks/asm/learned_rules_qwen12.json"); done
fi

prefix="${QUICK:+quick_}"
for item in "${schedule[@]}"; do
  seed="${item%%:*}" arm="${item#*:}"
  name="${arm%%=*}" rules="${arm#*=}"
  run="asm_v3_${name}_s${seed}"
  # The untrained "before" report depends only on the rule set: reuse seed 0's.
  first="checkpoints/${prefix}asm_v3_${name}_s${SEEDS%% *}"
  here="checkpoints/${prefix}${run}"
  if [[ "$here" != "$first" && -f "$first/report_before.txt" ]]; then
    mkdir -p "$here"
    cp -n "$first/report_before.txt" "$first/eval_before.csv" "$here/" 2>/dev/null || true
  fi
  RLGES_ASM_LEARNED="$rules" RUN="$run" SEED="$seed" scripts/overnight.sh
  # Refresh the table after every run, so a partial night still has one.
  specs=()
  for a in "${arms[@]}"; do specs+=("${a%%=*}=checkpoints/${prefix}asm_v3_${a%%=*}_s*"); done
  .venv/bin/python scripts/ablation_table.py "${specs[@]}" > "checkpoints/${prefix}ablation_v3.md" || true
done

specs=()
for arm in "${arms[@]}"; do
  name="${arm%%=*}"
  specs+=("$name=checkpoints/${prefix}asm_v3_${name}_s*")
done
.venv/bin/python scripts/ablation_table.py "${specs[@]}" | tee "checkpoints/${prefix}ablation_v3.md"
