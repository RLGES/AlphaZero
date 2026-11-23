#!/usr/bin/env bash
# Build, measure and select the e-graph test suites (docs/paper/TESTSET_SPEC.md).
# Resumable: measurement skips ids already in measured_/builtin_ files.
set -euo pipefail
cd "$(dirname "$0")/.."
W="${WORKERS:-2}"
D=benchmarks/asm/testsets
Q=benchmarks/asm/learned_rules_qwen_all.json
export PYTHONPATH=.
start=$(date +%s)
[[ -f $D/candidates_T7.jsonl ]] || RLGES_ASM_LEARNED=$Q nice -n 15 .venv/bin/python scripts/build_testsets.py candidates
echo "=== $(date '+%F %T') candidates ready ($(( $(date +%s) - start ))s)"
for s in T3 T4 T5 T7; do
  t=$(date +%s)
  RLGES_ASM_LEARNED=$Q nice -n 15 .venv/bin/python scripts/testset_stats.py measure $D/candidates_$s.jsonl $D/measured_$s.jsonl "$W" 2>&1 | grep -v "memory alloc\|RUST_BACK" | tail -1
  RLGES_ASM_LEARNED=none nice -n 15 .venv/bin/python scripts/testset_stats.py builtin $D/candidates_$s.jsonl $D/builtin_$s.jsonl "$W" 2>&1 | grep -v "memory alloc\|RUST_BACK" | tail -1
  echo "=== $(date '+%F %T') $s measured in $(( $(date +%s) - t ))s"
done
RLGES_ASM_LEARNED=$Q nice -n 15 .venv/bin/python scripts/build_testsets.py select
echo "=== $(date '+%F %T') test suites done in $(( $(date +%s) - start ))s"
