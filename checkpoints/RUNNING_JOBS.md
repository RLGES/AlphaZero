# Jobs running on the server (updated 2026-11-24 03:58:50 UTC)

## T3 baselines  (name: "T3 baselines")
- What: egglog default vs BackOff-200 vs MCTS-GEB vs untrained AZ-GEB on all 366 T3 large blocks,
  e-node limit 1000, then T3 at 300 nodes, then 150 terms each of T4, T5, T7.
- Started: 2026-11-23 20:17 UTC (scripts/run_geb_probes.sh, 3 workers).
- Main PID: 3482835; queue script PID: 3482831
- Watch: tail checkpoints/T3_BASELINES_RUNNING.log (= geb_probes.log); per-suite results land in
  checkpoints/geb_probe_<suite>_all_<limit>.json / .log
