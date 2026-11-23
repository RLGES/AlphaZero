#!/usr/bin/env bash
# Training-side rule-sensitive pool for T7-style training terms: generated train-split
# terms (seed 8888, depth 4-7) where egglog with the qwen rules beats built-in egglog.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
D=benchmarks/asm/testsets
W="${WORKERS:-3}"
.venv/bin/python - <<'PY'
import json
from random import Random
from benchmarks.generators import generate
from benchmarks.asm.term import to_sexpr
terms = generate("asm", Random(8_888), 1500, (4, 7), split="train")
with open("benchmarks/asm/testsets/train_candidates.jsonl", "w") as h:
    for i, t in enumerate(terms):
        h.write(json.dumps({"id": f"tr-{i:04d}", "sexpr": to_sexpr(t)}) + "\n")
PY
RLGES_ASM_LEARNED=none nice -n 10 .venv/bin/python scripts/testset_stats.py builtin $D/train_candidates.jsonl $D/train_builtin.jsonl "$W" 2>&1 | grep -v "memory alloc\|RUST" | tail -1
RLGES_ASM_LEARNED=benchmarks/asm/learned_rules_qwen_all.json nice -n 10 .venv/bin/python scripts/testset_stats.py builtin $D/train_candidates.jsonl $D/train_qwen.jsonl "$W" 2>&1 | grep -v "memory alloc\|RUST" | tail -1
.venv/bin/python - <<'PY'
import json
d = "benchmarks/asm/testsets/"
b = {json.loads(x)["id"]: json.loads(x)["egglog_builtin_cost"] for x in open(d + "train_builtin.jsonl")}
q = {json.loads(x)["id"]: json.loads(x)["egglog_builtin_cost"] for x in open(d + "train_qwen.jsonl")}
cands = {json.loads(x)["id"]: json.loads(x) for x in open(d + "train_candidates.jsonl")}
keep = [cands[i] for i in cands if b.get(i) is not None and q.get(i) is not None and q[i] < b[i]]
with open(d + "train_sensitive.jsonl", "w") as h:
    for r in keep:
        h.write(json.dumps(r) + "\n")
print(f"train-side rule-sensitive pool: {len(keep)} of {len(cands)}")
PY
