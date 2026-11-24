# Jobs running on the server (updated 2026-11-24 04:17:40 UTC)

## T3 baselines (name: "T3 baselines")
- What: egglog default vs BackOff-200 vs MCTS-GEB-style search vs untrained AZ-GEB.
- Restarted 2026-11-24 04:2x UTC on a fixed random sample of 100 T3 terms (seed 20261124), 5-min limit
  per method, saving after every term. The first run (8 h, all 366 terms) saved nothing and was stopped.
- Queue: scripts/run_geb_probes.sh -> T3 100@1000 nodes, T3 100@300, T4 50, T5 29, T7 50 (3 workers).
- Watch: checkpoints/T3_BASELINES_RUNNING.log (= geb_probes.log); results checkpoints/geb_probe_<suite>_all_<limit>.jsonl

## MCTS-GEB reproduction (name: "MCTS-GEB reproduction")
- Their released code: /root/rlges/external/eqs (venv /root/rlges/external/venv).
- Running: run_mcts.py --lang MATH --depth_lim 5 --seed 0 --node_lim 10000 --m wu --w2 4 --budget 512
- Log: /root/rlges/external/results/mcts_MATH-5-0.log (their paper: egg 169, MCTS-GEB 123)
