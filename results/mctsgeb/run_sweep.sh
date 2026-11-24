#!/usr/bin/env bash
# MCTS-GEB reproduction sweep with their released code: for each language and seed (depth-5
# expressions from their generator), run their egg baseline and their MCTS-GEB (WU-UCT,
# budget 512, 2 simulation workers) at the given e-node limit. Results: results/sweep_<limit>.csv
set -uo pipefail
cd /root/rlges/external/eqs/pes
PY=/root/rlges/external/venv/bin/python
LIMIT="${1:-500}"; SEEDS="${2:-20}"
OUT=/root/rlges/external/results/sweep_${LIMIT}.csv
[ -f "$OUT" ] || echo "lang,seed,limit,base_cost,egg_cost,egg_seconds,mcts_cost,mcts_seconds" > "$OUT"
for lang in MATH PROP; do
  for seed in $(seq 0 $((SEEDS - 1))); do
    grep -q "^$lang,$seed,$LIMIT," "$OUT" && continue
    t=$(date +%s.%N)
    egg=$(timeout 600 $PY pes/examples/run_egg.py --lang $lang --depth_lim 5 --seed $seed --node_lim $LIMIT --l 0 2>&1)
    te=$(echo "$(date +%s.%N) - $t" | bc)
    base=$(echo "$egg" | grep "base cost" | awk "{print \$NF}")
    eggc=$(echo "$egg" | grep "greedy cost" | awk "{print \$NF}")
    t=$(date +%s.%N)
    mcts=$(timeout 3600 nice -n 5 $PY pes/examples/run_mcts.py --lang $lang --depth_lim 5 --seed $seed --node_lim $LIMIT --m wu --env_id pes-par-env-v0 --w1 1 --w2 2 --budget 512 --l 0 2>&1)
    tm=$(echo "$(date +%s.%N) - $t" | bc)
    mc=$(echo "$mcts" | grep -o "extracted ast size [0-9.]*" | awk "{print \$NF}")
    echo "$lang,$seed,$LIMIT,$base,$eggc,$te,${mc:-NA},$tm" >> "$OUT"
    echo "$(date +%T) $lang seed $seed: base $base egg $eggc mcts ${mc:-NA}"
  done
done
echo "=== sweep $LIMIT done"
