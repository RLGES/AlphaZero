#!/usr/bin/env bash
# Run 4, step 2: iterated qwen rule discovery, rounds 1-4.
#
#   scripts/run4_discover.sh   # needs the LLM at RLGES_LLM_URL (default localhost:1234)
#
# Round 1 starts from the built-in rules; every later round is seeded from the terms
# still stuck with all earlier rules loaded, and counts those rules as duplicates, so
# it can only add new ones (this is how round 2 found the night-3 gain). Rules are
# random-tested and Z3-proved by discover.py. Finished rounds are kept on a rerun.
# Writes learned_rules_qwen{,2,3,4}.json and the merged learned_rules_qwen12.json,
# learned_rules_qwen123.json and learned_rules_qwen_all.json.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
D=benchmarks/asm
URL="${RLGES_LLM_URL:-http://localhost:1234/v1}"
MIN_NEW="${MIN_NEW:-3}"   # stop iterating once a round adds fewer rules than this

wait_for_llm() {
  until "$PY" -c "import urllib.request; urllib.request.urlopen('$URL/models', timeout=5)" 2> /dev/null; do
    echo "$(date '+%F %T') waiting for the LLM at $URL"
    sleep 120
  done
}

files=("$D/learned_rules_qwen.json")
merged_name=(none "$D/learned_rules_qwen.json" "$D/learned_rules_qwen12.json" "$D/learned_rules_qwen123.json")
prompts=(150 200 200 200)

for round in 1 2 3 4; do
  out="$D/learned_rules_qwen${round#1}.json"
  [[ $round == 1 ]] && out="$D/learned_rules_qwen.json"
  loaded="${merged_name[$((round - 1))]}"
  if [[ ! -f "$out" ]]; then
    known=()
    for f in "${files[@]:0:$((round - 1))}"; do known+=(--known "$f"); done
    wait_for_llm
    echo "=== $(date '+%F %T') round $round: rules loaded $loaded, ${prompts[$((round - 1))]} prompts"
    RLGES_ASM_LEARNED="$loaded" "$PY" -m benchmarks.asm.discover \
      --prompts "${prompts[$((round - 1))]}" --workers 2 --seed "$((round - 1))" \
      ${known[@]+"${known[@]}"} --out "$out.tmp" > "checkpoints/discover${round}.log" 2>&1
    mv "$out.tmp" "$out"
  fi
  [[ $round -gt 1 ]] && files+=("$out")
  new=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['accepted']))" "$out")
  echo "round $round: $new rules loaded"
  case $round in
    2) "$PY" scripts/merge_rules.py --out "$D/learned_rules_qwen12.json" "${files[@]}" ;;
    3) "$PY" scripts/merge_rules.py --out "$D/learned_rules_qwen123.json" "${files[@]}" ;;
  esac
  "$PY" scripts/merge_rules.py --out "$D/learned_rules_qwen_all.json" "${files[@]}"
  if (( round > 1 && new < MIN_NEW )); then
    echo "round $round added only $new rules: stopping discovery"
    break
  fi
done
echo "=== $(date '+%F %T') discovery done: $(grep -c '"lhs"' "$D/learned_rules_qwen_all.json") rules in learned_rules_qwen_all.json"
