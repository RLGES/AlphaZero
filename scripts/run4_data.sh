#!/usr/bin/env bash
# Run 4, step 1: rebuild the Learn-R corpora and stress set with the night-3 settings.
#
#   scripts/run4_data.sh       # needs a Learn-R checkout (LEARNR_PATH); rerun to redo all
#
# Same commands and seeds as the original commits (e440fd5, 76a442d, 9179943), so the
# terms match night 3 as long as Learn-R and the generators are unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
WORKERS="${WORKERS:-7}"
LEARNR="${LEARNR_PATH:-$HOME/Projects/RLGES/Learn-R}"
mkdir -p checkpoints
bridge() { "$PY" -m benchmarks.asm.learnr_bridge --learnr "$LEARNR" --style hard --hard-only "$@"; }

echo "=== $(date '+%F %T') hard corpus (night 3: 10,999 train / 911 test)"
bridge --workers "$WORKERS" --train-programs 3000 --test-programs 300 > checkpoints/hard_corpus.log 2>&1

echo "=== $(date '+%F %T') control-flow corpus (night 3: 8,172 / 811 after dedup)"
bridge --control-flow 0.35 --train-programs 800 --test-programs 100 --prefix cf_ \
  --workers "$WORKERS" > checkpoints/cf_corpus.log 2>&1

# Keep train and test disjoint across the two corpora.
"$PY" - <<'EOF'
import json
p = lambda n: f"benchmarks/asm/corpus/{n}.jsonl"
load = lambda n: list(open(p(n)))
key = lambda line: json.loads(line)["sexpr"]
train = {key(l) for l in load("train")}
test = {key(l) for l in load("test")}
cf_test = [l for l in load("cf_test") if key(l) not in train | test]
cf_test_keys = {key(l) for l in cf_test}
cf_train = [l for l in load("cf_train") if key(l) not in test | cf_test_keys]
for name, rows in [("cf_test", cf_test), ("cf_train", cf_train)]:
    open(p(name), "w").writelines(rows)
    print(name, len(rows))
EOF

echo "=== $(date '+%F %T') stress set (night 3: 824 terms, 784 after corpus dedup)"
bridge --statements 15,28 --max-nodes 300 --control-flow 0.25 --train-programs 0 \
  --test-programs 50 --seed 2026 --prefix stress_ --workers "$WORKERS" > checkpoints/stress_corpus.log 2>&1
rm -f benchmarks/asm/corpus/stress_train.jsonl

wc -l benchmarks/asm/corpus/*.jsonl
echo "=== $(date '+%F %T') done"
