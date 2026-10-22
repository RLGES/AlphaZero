#!/usr/bin/env bash
# Overnight pipeline: Learn-R -> asm suite -> AlphaZero training -> reports.
#
#   scripts/overnight.sh            # full night (keep the laptop open and plugged in)
#   QUICK=1 scripts/overnight.sh    # 2-minute smoke run of every step
#
# Steps (each one is skipped or resumed if it already finished):
#   1. check the environment and the Learn-R checkout (LEARNR_PATH)
#   2. only with REEXPORT=1 on a fresh run: re-export the classic Learn-R corpus
#      and re-import Learn-R's learned rules (the committed hard corpus is kept
#      otherwise). The rule set comes from RLGES_ASM_LEARNED (none|<path>).
#   3. run the test suite; stop if anything fails
#   4. "before" report: untrained AlphaZero vs MCTS on held-out Learn-R terms
#   5. train AlphaZero on asm (Learn-R rules, Learn-R-compiled programs),
#      early-stopping on a hard held-out score; resumes from latest.pt
#   6. "after" report with the best network
#   (SKIP_ASM=1 skips steps 4-6)
#   7. with time left, $EXTRA_TAG runs (with an after report) for EXTRA_SUITES (default: boolean bitvector arithmetic)
# Logs: checkpoints/overnight.log; morning summary: checkpoints/summary_<RUN>.txt
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LEARNR_PATH="${LEARNR_PATH:-$HOME/Projects/RLGES/Learn-R}"
EXTRA_SUITES="${EXTRA_SUITES:-boolean bitvector arithmetic}"
RUN="${RUN:-asm_v2}"
EXTRA_TAG="${EXTRA_TAG:-v3}"
WORKERS="${WORKERS:-7}"
if [[ -n "${QUICK:-}" ]]; then
  RUN="quick_${RUN}"
  TRAIN=(--rounds 2 --games 8 --sims 8 --max-moves 10 --steps 5 --hidden 32 --layers 1
         --heldout 5 --eval-iterations 10 --patience 0)
  EVAL=(--budgets 5 --seeds 1 --generated 5)
  EXTRA_TRAIN=(--rounds 1 --games 4 --sims 4 --steps 2 --hidden 16 --layers 1 --heldout 3
               --eval-iterations 5)
else
  TRAIN=(--rounds 100 --games 256 --sims 96 --max-moves 40 --depth 3,6 --steps 400
         --batch-size 128 --hidden 192 --layers 5 --heldout 100 --eval-iterations 25 --patience 12)
  EVAL=(--budgets 10,25,50,100 --seeds 2 --generated 50)
  EXTRA_TRAIN=(--rounds 60 --games 256 --sims 128 --max-moves 40 --depth 3,7 --steps 400
               --batch-size 128 --hidden 192 --layers 5 --heldout 100 --eval-iterations 25 --patience 8)
fi
OUT="checkpoints/$RUN"
SUMMARY="checkpoints/summary_$RUN.txt"
mkdir -p "$OUT"
LOG=checkpoints/overnight.log
exec > >(tee -a "$LOG") 2>&1
caffeinate -i -s -w $$ &  # no idle or system sleep while this script runs

step() { echo; echo "=== $(date '+%F %T') $*"; }

# Each round writes ~5 MB of checkpoints; stop cleanly instead of filling the disk.
disk_ok() {
  local free_kb
  free_kb=$(df -k . | awk 'NR==2 {print $4}')
  if (( free_kb < ${MIN_FREE_GB:-2} * 1024 * 1024 )); then
    echo "STOP: only $((free_kb / 1024)) MB free; free some space and rerun to resume"
    return 1
  fi
}

step "1. environment"
"$PY" -c "import torch, egglog; print('torch', torch.__version__)"
if [[ -d "$LEARNR_PATH/frontend" ]]; then
  echo "Learn-R: $LEARNR_PATH ($(git -C "$LEARNR_PATH" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"
else
  echo "Learn-R not found at $LEARNR_PATH: using the committed corpus and rules"
  LEARNR_PATH=""
fi

step "2. Learn-R corpus and learned rules"
echo "rule set: RLGES_ASM_LEARNED=${RLGES_ASM_LEARNED:-<default learned_rules.json>}"
if [[ -f "$OUT/latest.pt" ]]; then
  echo "training already started: keeping the current corpus and rule list"
elif [[ -z "${REEXPORT:-}" ]]; then
  echo "keeping the committed corpus ($(wc -l < benchmarks/asm/corpus/train.jsonl) train terms)"
elif [[ -n "$LEARNR_PATH" ]]; then
  LEARNR_PATH="$LEARNR_PATH" "$PY" -m benchmarks.asm.learnr_bridge --learnr "$LEARNR_PATH"
  "$PY" - "$LEARNR_PATH" <<'EOF'
import sys
from pathlib import Path
from benchmarks.asm.learned import import_rules
from benchmarks.asm.rules import PATTERN_RULES
root = Path(sys.argv[1])
paths = sorted(root.glob("output*.txt")) + sorted(root.rglob("learned_rules_db.json"))
report = import_rules(paths, {(lhs, rhs) for _, lhs, rhs in PATTERN_RULES})
print(f"learned rules: {len(report['accepted'])} accepted, {len(report['rejected'])} rejected")
EOF
fi

step "3. tests"
"$PY" -m pytest -q -x 2>&1 | tail -3

if [[ -z "${SKIP_ASM:-}" ]]; then  # SKIP_ASM=1: only the extra suites
  step "4. before: untrained AlphaZero vs MCTS on held-out Learn-R terms"
  if [[ ! -f "$OUT/report_before.txt" ]]; then
    "$PY" -m src.engines.alphazero.evaluate --suite asm "${EVAL[@]}" --no-egglog \
      --csv "$OUT/eval_before.csv" | tee "$OUT/report_before.txt"
  else
    cat "$OUT/report_before.txt"
  fi

  step "5. train AlphaZero on asm -> $OUT"
  disk_ok
  resume=()
  [[ -f "$OUT/latest.pt" ]] && resume=(--resume)
  "$PY" -m src.engines.alphazero.train --suite asm "${TRAIN[@]}" --workers "$WORKERS" \
    --out "$OUT" ${resume[@]+"${resume[@]}"} 2>&1 | tee -a "$OUT/train.log"

  step "6. after: trained AlphaZero (best.pt) vs MCTS"
  "$PY" -m src.engines.alphazero.evaluate --suite asm --checkpoint "$OUT/best.pt" "${EVAL[@]}" \
    --no-egglog --csv "$OUT/eval_after.csv" | tee "$OUT/report_after.txt"

  {
    echo "Overnight summary: $RUN, rules ${RLGES_ASM_LEARNED:-default} ($(date '+%F %T'))"
    echo; echo "asm training (last rounds):"; tail -5 "$OUT/train.log"
    echo; echo "BEFORE (untrained):"; cat "$OUT/report_before.txt"
    echo; echo "AFTER (best.pt):"; cat "$OUT/report_after.txt"
  } > "$SUMMARY"

fi

step "7. extra $EXTRA_TAG runs: $EXTRA_SUITES"
for suite in $EXTRA_SUITES; do
  out="checkpoints/${QUICK:+quick_}${suite}_${EXTRA_TAG}"
  mkdir -p "$out"
  resume=()
  [[ -f "$out/latest.pt" ]] && resume=(--resume)
  step "   $suite -> $out"
  disk_ok || break
  "$PY" -m src.engines.alphazero.train --suite "$suite" "${EXTRA_TRAIN[@]}" --workers "$WORKERS" \
    --out "$out" ${resume[@]+"${resume[@]}"} 2>&1 | tee -a "$out/train.log"
  "$PY" -m src.engines.alphazero.evaluate --suite "$suite" --checkpoint "$out/best.pt" "${EVAL[@]}" \
    --no-egglog --csv "$out/eval_after.csv" | tee "$out/report_after.txt"
  {
    echo; echo "$suite $EXTRA_TAG (last rounds):"; tail -3 "$out/train.log"
    echo "AFTER (best.pt):"; cat "$out/report_after.txt"
  } >> "$SUMMARY"
done

step "done. Summary: $SUMMARY"
