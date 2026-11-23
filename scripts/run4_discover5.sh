#!/usr/bin/env bash
# Run 4, extra discovery round 5: seeded from terms still stuck with all 80 rules
# loaded, every earlier round counted as known. Writes learned_rules_qwen5.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=benchmarks/asm
if [[ -f checkpoints/llm.env ]]; then set -a; source checkpoints/llm.env; set +a; fi
out="$D/learned_rules_qwen5.json"
[[ -f "$out" ]] && { echo "round 5 already done"; exit 0; }
echo "=== $(date '+%F %T') round 5: rules loaded $D/learned_rules_qwen_all.json, 200 prompts"
RLGES_ASM_LEARNED="$D/learned_rules_qwen_all.json" nice -n 10 "$PY" -m benchmarks.asm.discover \
  --prompts 200 --workers 2 --seed 4 \
  --known "$D/learned_rules_qwen.json" --known "$D/learned_rules_qwen2.json" \
  --known "$D/learned_rules_qwen3.json" --known "$D/learned_rules_qwen4.json" \
  --out "$out.tmp" > checkpoints/discover5.log 2>&1
mv "$out.tmp" "$out"
echo "=== $(date '+%F %T') round 5 done: $(grep -c . /dev/null; "$PY" -c "import json;d=json.load(open('$out'));print(len(d['accepted']),'loaded,',len(d['verified_not_loaded']),'more verified')")"
