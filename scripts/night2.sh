#!/usr/bin/env bash
# Night 2: the rule-set ablation on the hard Learn-R corpus, then the other suites.
#
#   scripts/night2.sh          # keep the laptop open and plugged in; rerun to resume
#   QUICK=1 scripts/night2.sh  # smoke run of all three stages
#
# 1. asm_v2      Learn-R's 44 built-in rules (+ fold) only
# 2. asm_v2_llm  built-in + LLM-discovered rules (skipped if LLM_RULES has none)
# 3. boolean / bitvector / arithmetic v3 runs, as time allows
# Summaries: checkpoints/summary_<run>.txt
set -euo pipefail
cd "$(dirname "$0")/.."

LLM_RULES="${LLM_RULES:-benchmarks/asm/learned_rules_qwen.json}"

RLGES_ASM_LEARNED=none RUN=asm_v2 EXTRA_SUITES="" scripts/overnight.sh

if [[ -f "$LLM_RULES" ]] && .venv/bin/python -c "
import json, sys
sys.exit(0 if json.load(open('$LLM_RULES')).get('accepted') else 1)"; then
  RLGES_ASM_LEARNED="$LLM_RULES" RUN=asm_v2_llm EXTRA_SUITES="" scripts/overnight.sh
else
  echo "no accepted rules in $LLM_RULES: skipping the asm_v2_llm arm"
fi

RLGES_ASM_LEARNED=none RUN=asm_v2 SKIP_ASM=1 scripts/overnight.sh
