#!/usr/bin/env bash
# Run 4: the night-3 experiment again, aiming higher. Run after scripts/run4_data.sh.
#
#   scripts/run4.sh    # rerun to resume: finished runs and rounds are skipped
#
# 1. qwen discovery rounds 1-4 (scripts/run4_discover.sh) in the background, while
#    the builtin arm trains (3 seeds); discovery needs the LLM, not much CPU.
# 2. llm arm: built-in + every discovered rule (learned_rules_qwen_all.json), 3 seeds.
# 3. llmft arm: continue each llm seed from its best.pt with a lower learning rate and
#    twice the self-play simulations, to get past the round-35-45 plateau of night 3.
# 4. final evaluation of every arm on all held-out and stress terms (as night 3).
# 5. llmbig: a larger network (256 hidden, 6 layers) on every rule, seed 0, then the
#    final evaluation again (finished arms are not recomputed).
# Same training settings as night 3 otherwise (HELDOUT 300, PATIENCE 15, GENERATED 150).
# Runs: checkpoints/asm_v4_<arm>_s<seed>. Log: checkpoints/run4.log.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2}"
ALL=benchmarks/asm/learned_rules_qwen_all.json
export HELDOUT="${HELDOUT:-300}" PATIENCE="${PATIENCE:-15}" GENERATED="${GENERATED:-150}"
export EXTRA_SUITES=""
exec > >(tee -a checkpoints/run4.log) 2>&1

train() {  # train <arm> <rules> <seed> [VAR=value ...]
  local arm=$1 rules=$2 seed=$3
  shift 3
  local run="asm_v4_${arm}_s${seed}" first="checkpoints/asm_v4_${arm}_s${SEEDS%% *}"
  if [[ -f "checkpoints/$run/report_after.txt" ]]; then
    echo "$run: finished"
    return
  fi
  # The untrained "before" report depends only on the rule set: reuse the first seed's.
  if [[ "checkpoints/$run" != "$first" && -f "$first/report_before.txt" ]]; then
    mkdir -p "checkpoints/$run"
    cp -n "$first/report_before.txt" "$first/eval_before.csv" "checkpoints/$run/" 2> /dev/null || true
  fi
  env "$@" RLGES_ASM_LEARNED="$rules" RUN="$run" SEED="$seed" scripts/overnight.sh
}

table() {
  .venv/bin/python scripts/ablation_table.py builtin='checkpoints/asm_v4_builtin_s*' \
    llm='checkpoints/asm_v4_llm_s*' llmft='checkpoints/asm_v4_llmft_s*' \
    llmbig='checkpoints/asm_v4_llmbig_s*' \
    > checkpoints/ablation_v4.md 2> /dev/null || true
}

echo "=== $(date '+%F %T') run 4 start"
scripts/run4_discover.sh > checkpoints/run4_discover.log 2>&1 &
discover=$!

for seed in $SEEDS; do
  train builtin none "$seed" WORKERS=6
  table
done

echo "=== $(date '+%F %T') waiting for discovery (pid $discover)"
wait "$discover"
tail -3 checkpoints/run4_discover.log

for seed in $SEEDS; do
  train llm "$ALL" "$seed" WORKERS=7
  table
done

for seed in $SEEDS; do
  train llmft "$ALL" "$seed" WORKERS=7 INIT="checkpoints/asm_v4_llm_s${seed}/best.pt" \
    LR=3e-4 SIMS=192 ROUNDS=60
  table
done

echo "=== $(date '+%F %T') final evaluation"
TAG=v4 ARMS="builtin=none llm=$ALL llmft=$ALL" scripts/final_eval.sh

train llmbig "$ALL" "${SEEDS%% *}" WORKERS=7 HIDDEN=256 LAYERS=6
table
echo "=== $(date '+%F %T') final evaluation with llmbig"
TAG=v4 ARMS="builtin=none llm=$ALL llmft=$ALL llmbig=$ALL" scripts/final_eval.sh
echo "=== $(date '+%F %T') run 4 done"
