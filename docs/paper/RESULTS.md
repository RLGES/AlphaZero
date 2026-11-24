# Run 4 results log (for the paper)

> Terminology: "BackOff-200" = egglog with the back-off scheduler at match limit 200 (ban length 2, up to 12 iterations); "egglog default" = back-off scheduler at match limit 1000, 6 iterations. Earlier notes called BackOff-200 "capped" / "all rules capped".

Every number here comes from a file in this repository (path given) produced on the
run-4 server. Updated as results arrive. "Improvement" = mean relative cost reduction
over terms, in percent. Budgets are search iterations. Bold = headline result.

## Setup
- Suite: asm (Learn-R IR), cost = op latencies (mul 3, div/mod 10, load 4, other 1).
- Test sets (rebuilt with the night-3 commands and seeds): 1,722 held-out Learn-R
  terms and 784 stress terms. Check: built-in MCTS@100 on held-out = 66.1, identical
  to night 3 (`checkpoints/final_mcts_none_heldout.csv`).
- Training: AlphaZero (PUCT + GNN), early stop after 15 rounds without a new best
  on 300 generated validation terms (score = mean improvement at 25 iterations).
- Hardware: 8 vCPU EPYC, 15 GiB RAM, no GPU.

## 1. Rule discovery (qwen2.5-coder-7b via Ollama, Z3-verified)
Logs: `checkpoints/discover{1..5}.log`, rules `benchmarks/asm/learned_rules_qwen*.json`.

| Round | Prompts | Verified | Loaded |
|---|---|---|---|
| 1 | 150 | 105 | 20 |
| 2 | 200 | 89 | 20 |
| 3 | 200 | 71 | 20 |
| 4 | 200 | 76 | 20 |
| 5 (extra) | 200 | 66 | 20 |

- 80 rules (rounds 1-4) = `learned_rules_qwen_all.json`, used for all llm runs.
- Rule-set selection check, plain MCTS on 300 train-distribution terms (seed 424242,
  `scripts/rule_select_eval.py`, `checkpoints/rulecheck5.log`):

| Rule set | MCTS@25 | MCTS@100 |
|---|---|---|
| built-in only | 57.4 | 59.4 |
| qwen_all (80) | 65.2 | 66.7 |
| all 189 verified cost-lowering (rounds 1-4) | 65.2 | 66.9 |
| qwen_all + round 5 (100) | 65.4 | 66.8 |
| all 222 verified cost-lowering (rounds 1-5) | 65.3 | 67.0 |

Finding: discovery saturates; the top-20-per-round filter is not the bottleneck.
Round-5 rules almost never match terms that are still stuck (seed_matches ~0).

## 2. Term-level AlphaZero training
`checkpoints/asm_v4_<arm>_s<seed>/train.log`, `metrics.jsonl`.

| Run | Rounds | Best validation score |
|---|---|---|
| builtin s0 / s1 / s2 | 55 / 32 / 55 | 0.552 / 0.544 / 0.560 |
| llm s0 / s1 | 89 / 47 | 0.646 / 0.636 |
| llm s2 | (running) | 0.632 at round 43 |

## 3. AlphaZero vs MCTS, 314 held-out terms (per-run reports)
`checkpoints/asm_v4_*/report_before.txt`, `report_after.txt`. Columns: improvement,
best-known share, transitions (calls to apply).

Built-in rules, trained (s0 / s1 / s2), AZ vs MCTS:

| Budget | AZ s0 | AZ s1 | AZ s2 | MCTS |
|---|---|---|---|---|
| 10 | 50.6 | 49.4 | 50.9 | 54.4 |
| 25 | 56.0 | 55.2 | 56.4 | 58.0 |
| 50 | 58.3 | 58.2 | 58.6 | 59.1 |
| 100 | 59.8 | 60.1 | 59.6 | 59.9 |

Best-known @100: AZ 83/85/85% vs MCTS 80-81%. Transitions @100: AZ ~89 vs MCTS 583
(6.5x fewer). Untrained AZ @100: 59.2 (best-known 78%).

qwen rules, trained (s0 / s1), AZ vs MCTS:

| Budget | AZ s0 | AZ s1 | MCTS |
|---|---|---|---|
| 10 | 60.5 | 58.8 | 62.5 |
| 25 | 65.0 | 64.0 | 65.2 |
| 50 | 66.6 | 66.0 | 66.0 |
| 100 | **67.1** | **67.3** | 66.5 |

Best-known @100: AZ 89/90% vs MCTS 82-83%. Transitions @100: AZ 80-83 vs MCTS 622
(~7.8x fewer). Untrained AZ (qwen rules) @100: 66.6.

## 4. Final test sets, engines that need no training
`checkpoints/final_{mcts,egglog,egglogmcts}_<rules>_<set>.csv`.

| Engine | Held-out, built-in | Held-out, qwen | Stress, built-in | Stress, qwen |
|---|---|---|---|---|
| MCTS @10 | 56.3 | 66.7 | 51.3 | 61.6 |
| MCTS @25 | 63.0 | 71.8 | 56.6 | 66.0 |
| MCTS @50 | 64.6 | 73.1 | 58.6 | 67.4 |
| MCTS @100 | 66.1 | 74.0 | 60.4 | 68.4 |
| egglog (6 iterations) | 80.5 | 80.8 | 75.6 | 75.7 |
| egglog + MCTS (any budget) | 80.6 | 80.9 | 75.7 | 75.8 |

- Rule lift for MCTS@100 on held-out: +7.9 (night 3's rules: +12.5).
- Rule lift for egglog: +0.3. Discovered rules matter for step-wise search, barely for saturation.
- Time per term, held-out: egglog 0.13 s (built-in) / 0.64 s (qwen, max 7.2 s); MCTS@100 3.3 s.
- **egglog memory blow-up with many rules:** with the 80 qwen rules one held-out term
  reached 11.7 GB and was OOM-killed. Final eval now runs egglog under a 2 GB cap,
  retrying with fewer iterations: 30 terms needed it (26 at 4 iterations, 4 at 2), none
  failed (`checkpoints/egglog_fallbacks.log`). Built-in rules: no cap needed.
- Only 79 of 1,722 held-out terms change under egglog when the qwen rules are added
  (75 better, 4 worse).

## 5. AlphaZero-first hybrids (no new training)
`checkpoints/asm_v4_llm_s<seed>/final_<set>_{azegg,azrules}.csv`.

60-term pilot (llm s0, @100): egglog all 125 rules 86.9 (4,100 rule matches), egglog
built-in 86.3, AZ then egglog 86.6; egglog on rules AZ picks from ALL rules k=24/40/60:
79.1/79.5/81.4; built-in + AZ top-k learned + AZ path: k=0 86.7, **k=10 87.1 (2,643
matches, -35%)**, k=20 87.1.

Full sets, llm s0, k=10:

| Hybrid | Held-out @10/25/50/100 | Stress @10/25/50/100 |
|---|---|---|
| AZ then egglog | 80.69 / 80.65 / 80.68 / 80.74 | 75.76 / 75.73 / 75.81 / 75.82 |
| egglog on AZ-chosen rules | 80.73 / 80.76 / 80.78 / 80.79 | 75.67 / 75.68 / 75.68 / 75.70 |

Finding: on the full sets both hybrids tie egglog with all rules (80.8 / 75.7); the
pilot's +0.2 did not hold. Seeds 1-2: running.

## 6. AlphaZero on e-graphs (AZ-GEB), in progress
Design: `docs/run4/EGRAPH_PLAN.md`; code `benchmarks/asm/geb.py`, `scripts/geb_train.py`.

Untrained smoke tests (`scripts/geb_smoke.py`):
- MCTS-GEB action space (one rule, one iteration), 10 held-out terms, 2,000 nodes:
  egglog 81.5 (0.3 s), MCTS-GEB 62.8 (11 s), AZ-GEB untrained 44.6 (5.5 s).
- Guided saturation (enable one learned rule, then one iteration of built-in + enabled),
  20 rule-sensitive terms: 300 nodes: egglog 77.9 / MCTS-GEB 77.4 / AZ-GEB 75.2;
  1,000 nodes: 79.3 / 78.5 / 76.5.
- Training run `checkpoints/geb_v1` (guided saturation, init llm s0, 1,000 nodes, 8 steps,
  16 sims, 32 episodes/round): validation (40 train-distribution terms, depth 4-6, seed
  515151) flat at 69.49 for 11 rounds, early-stopped; e-graphs ~116 nodes (the 8-step
  limit binds, not the node limit).
- `checkpoints/geb_v2` (enable any rule, starting from folding only; 400 nodes, 16 steps):
  ~36 vs egglog 70 with 24-node e-graphs; enabling one rule per step is far too slow.
  Stopped by hand at round 8.
- Ablation on the same 40 validation terms, 1,000 nodes, 8 steps (`/tmp/geb_ablate.py`,
  copied to `scripts/geb_ablate.py`):

| Policy | Improvement | Mean e-nodes |
|---|---|---|
| egglog default schedule (6 iterations, match limit 1000) | 70.17 | 580 |
| built-in rules only, match limit 200 per step, 8 steps | 68.45 | 115 |
| **all rules enabled, match limit 200 per step, 8 steps** | **71.69** | **211** |
| AZ-GEB v1 (learned choice of rules to enable) | 69.49 | 118 |

  Finding: a tighter per-step match limit with more steps beats egglog's default
  schedule with ~2.7x fewer e-nodes; learning which learned rules to enable does not
  beat enabling them all. The next design starts from all rules and learns what to prune.

## 7. FINAL run-4 tables (final eval finished 23 Nov 18:32 UTC)
Sources: `checkpoints/final_v4_heldout.md`, `final_v4_stress.md`, `final_v4_*_egglog.md`
(3 seeds, mean ± std over seeds, paired per-term bootstrap 95% CI).

Held-out (1,722), trained AZ vs MCTS:

| Arm | Budget | AZ | MCTS | AZ − MCTS | AZ Δ vs builtin (95% CI) |
|---|---|---|---|---|---|
| builtin | 10 | 53.6 ± 1.3 | 56.3 | -2.7 | — |
| builtin | 100 | 66.3 ± 0.3 | 66.1 | +0.1 | — |
| qwen | 10 | 64.4 ± 1.1 | 66.7 | -2.3 | +10.8 [+9.7, +11.9] |
| qwen | 25 | 71.2 ± 0.7 | 71.8 | -0.6 | +10.4 [+9.5, +11.6] |
| qwen | 50 | 73.7 ± 0.4 | 73.1 | +0.6 | +9.6 [+8.7, +10.6] |
| **qwen** | **100** | **75.0 ± 0.2** | 74.0 | **+0.9** | **+8.7 [+7.8, +9.7]** |

Stress (784): qwen @100 AZ 68.6 ± 0.2 vs MCTS 68.4 (+0.2); Δ vs builtin +9.2 [+7.8, +10.8];
@10 59.5 vs 61.6. Builtin @100 59.4 vs 60.4.

egglog comparison (held-out, qwen @100): AZ − egglog -5.8 [-6.5, -5.2]; egglog+AZ − egglog
+0.1 [+0.1, +0.1]; egglog+AZ − egglog+MCTS -0.0. AZ-first hybrids, 3 seeds: AZ→egglog
held-out 80.74/80.66/80.62, stress 75.82/75.66/75.68; egglog on AZ-chosen rules held-out
80.79/80.76/80.75 (`asm_v4_llm_s*/final_*_{azegg,azrules}.csv`).

314-term reports (`checkpoints/ablation_v4.md`): qwen @100 AZ 67.1 ± 0.2 vs MCTS 66.5.

## 8. Memory: egglog with all rules vs egglog on AlphaZero-chosen rules (measured)
`scripts/memory_compare.py`, `checkpoints/memory_compare_heldout.csv` (all 1,722 held-out
terms, llm s0 network, k=10). Peak RAM = ru_maxrss of each egglog child; "extra" subtracts
the trivial-term baseline (182.9 MB all rules, 179.5 MB AZ rules). 2 GB cap, retry with
4/2/1 iterations on failure.

| | egglog, all rules | egglog, AZ-chosen rules |
|---|---|---|
| Rules per term | 124 | 54.8 (44 built-in + 10 learned + AZ path) |
| Improvement | 80.83 | 80.79 |
| Extra RAM mean / median | 63.7 / 48.5 MB | 50.2 / 48.5 MB |
| Extra RAM p95 | 117.5 MB | 56.1 MB |
| Extra RAM max (successful run) | 931.2 MB | 96.4 MB |
| Terms over 100 MB extra | 112 | 0 |
| Terms exceeding the 2 GB cap (retried) | 11 | 0 |
| Rule matches (mean) | 3,769 | 2,090 (-45%) |
| Seconds per term (egglog only) | 0.711 | 0.486 (-32%) |

Finding: same quality (-0.04), with the memory tail removed: p95 halves, the worst case drops
from >2 GB (11 terms) to under 100 MB extra, and egglog runs 32% faster. The max for all
rules is understated: the 11 retried terms first exceeded 2 GB.

## 9. New e-graph test suites (built 24 Nov; spec docs/paper/TESTSET_SPEC.md)
`benchmarks/asm/testsets/<suite>.jsonl`, `<suite>_manifest.json` (per-term aspects),
`summary.json`; built by `scripts/build_testsets.py`, measured by `scripts/testset_stats.py`
(log `checkpoints/testsets.log`, 5,010 s total). Separation checked against run-4 training
terms (76,800 + validation), held-out + stress, and earlier suites.

| Suite | Terms | Buckets | Acceptance check |
|---|---|---|---|
| T3 large blocks | 366 | L 167, XL 199 | count PASS; L/XL >= 100 PASS; sharing 99.5% PASS; egglog fail or > 3k e-nodes 15.0% PASS (target 15%); egglog failed outright 7.7% |
| T4 explosion | 300 | L 234, M 66 | count PASS; AC chain >= 6 100% PASS; egglog fail or > 5k e-nodes **0.0% FAIL** (target 25%) |
| T5 deep / non-greedy | **29** | L 15, XL 10, M 4 | **count FAIL** (target 200): of 500 candidates, 259 duplicated held-out/stress and 82 training terms; non-greedy 100% of kept |
| T7 rule-sensitive | 249 | L 135, M 58, XL 55 | count PASS; rule-sensitive 100% PASS |

All suites: best-known outputs equivalent to inputs on 200 random 32-bit inputs (100%).
Findings: egglog's back-off scheduler keeps long same-operator chains under control (T4); the
asm generator's test split overlaps the existing held-out/stress sets heavily (T5), so T5 needs
deeper or differently seeded terms. XL is BackOff-200 at 256 nodes by the language.

## 10. AZ-GEB training on suite-style terms (24 Nov)
Training terms: T3/T4/T5/T7-style built from training-split material only
(`scripts/geb_train.py --composite -1`); validation 20 terms of the same kind (seed 515151).
Mixed action space: single rule (one BackOff-200 iteration) or macro (all rules, one BackOff-200
iteration); macro prior floor 0.5.

- geb_v3 (no macro floor, 1,000 nodes): round 0 AZ 21.3 vs egglog 67.5 vs BackOff-200 69.4 (the
  term network gives the macro action ~0 prior). Stopped.
- geb_v4/v5/v6: infrastructure failures (egglog memory growth in workers; Python
  max_tasks_per_child hang; > 2 h per round). Fixed with a fresh worker pool per batch.
- geb_v7 (500 nodes, 20 steps, 8 sims, 16 episodes): 9 rounds of ~6 min, flat:
  AZ 70.19 = BackOff-200 70.19 > egglog 65.70; e-nodes AZ 408 vs egglog 644. The policy collapsed
  to always taking the macro action (policy loss 0.02-0.04). Early-stopped at round 8.
  Finding: under a 500-node budget, BackOff-200 scheduling beats egglog's default by +4.5 with
  37% fewer e-nodes; learning has not yet improved on BackOff-200.
- geb_v8 started: 200-node budget (tighter, where one BackOff-200 step overshoots), 16 sims,
  macro floor 0.3.

### Memory comparison on every test set (llm s0 network, k=10; `checkpoints/memory_compare_<set>.csv`)
Extra RAM = peak RSS of the egglog child minus the trivial-term baseline (182.9 MB all rules,
179.5 MB AZ rules). "Retried" = runs that exceeded the 2 GB cap at 6 iterations.

| Set | Terms | Score all / AZ-rules | Extra RAM mean (all -> AZ) | p95 | max (successful) | Retried (all -> AZ) | Seconds (all -> AZ) |
|---|---|---|---|---|---|---|---|
| T1 held-out | 1722 | 80.83 / 80.79 | 63.7 -> 50.2 MB | 117.5 -> 56.1 | 931 -> 96 | 11 -> 0 | 0.71 -> 0.49 |
| T2 stress | 784 | 75.74 / 75.70 | 67.6 -> 62.0 | 92.5 -> 84.3 | 577 -> 97 | 6 -> 0 | 0.62 -> 0.47 |
| T3 large blocks | 366 | 89.05 / 88.50 | 72.6 -> 21.1 | 329.9 -> 28.0 | 888 -> 85 | 12 -> 0 | 1.32 -> 0.79 |
| T4 explosion | 300 | 37.88 / 37.36 | 15.2 -> 16.0 | 24.3 -> 25.0 | 38 -> 33 | 0 -> 0 | 1.05 -> 1.01 |
| T5 deep | 29 | 56.52 / 56.10 | 17.7 -> 9.3 | 72.5 -> 15.0 | 102 -> 28 | 0 -> 0 | 0.72 -> 0.53 |
| T7 rule-sensitive | 249 | 73.71 / 72.72 | 42.9 -> 16.0 | 173.9 -> 22.0 | 689 -> 38 | 3 -> 0 | 1.01 -> 0.64 |

Finding: across every set, egglog on AlphaZero-chosen rules never needed a retry (32 retries
-> 0), cut the memory tail sharply (T3 p95 330 -> 28 MB extra) and ran 25-40% faster, for a
score cost of 0.0-1.0 points (largest on T7, where the qwen rules matter most). T4 shows no
memory effect (egglog never blows up there).

## 11. Head-to-head on the MCTS-GEB benchmarks (their released code)
Code: github.com/ucamrl/eqs (commit bba5059, cloned on the server, run unmodified in a separate
Python 3.10 venv; Rust library built with maturin 0.13.7). Benchmarks: their Math (26 rules) and
Prop (13 rules) languages, depth-5 expressions from their generator, seeds 0-19 each, cost =
AstSize, e-node limit 500 (their scripts' default; 10k needed more than our 15 GB RAM: their
paper used 256 GB). Files: `results/mctsgeb/sweep_500.csv` (their egg baseline + MCTS-GEB),
`results/mctsgeb/egg_proper_500.csv` (egg Runner, all rules per iteration), scripts alongside.
Reproduction check: their egg baseline on MATH-5-0 at 10k nodes = 251 -> 169, as in their paper.

Mean relative size reduction (higher is better), terms where MCTS-GEB produced a result:

| Set | Terms | Their egg baseline (round-robin, 1 rule/step) | egg Runner, back-off | egglog default | egglog BackOff-200 | MCTS-GEB (original) |
|---|---|---|---|---|---|---|
| Math | 20 | 14.9% | 17.8% | 17.2% | 18.6% | **28.2%** |
| Prop | 18 | 45.5% | 45.9% | 48.1% | 48.4% | **69.5%** |

egglog columns: `scripts/egglog_mathprop.py`, `results/mctsgeb/egglog_500.csv`; same 40 expressions
(exported from their generator: `results/mctsgeb/mctsgeb_exprs.json`), their 26 + 13 rules
transcribed one-to-one, AstSize, no constant folding (their runs have none: `new_egraph`
passes no evaluator), strict 500-node limit (an iteration that would cross it is rolled
back). BackOff-200 is never better than MCTS-GEB (worse on 8/20 Math, 12/18 Prop).
Note: a first egglog run with constant folding and a soft limit (up to ~1,000 e-nodes)
looked much better (Prop 87%); that run was unfair and is discarded.

MCTS-GEB is strictly better than egg's back-off runner on 10/20 Math and 12/18 Prop terms and
never worse; ~51 s (Math) / ~73 s (Prop) per term with 2 simulation workers vs milliseconds
for egg. Two tiny Prop terms (seeds 12, 16) gave no MCTS-GEB output.
Finding: unlike our asm programs, these benchmarks leave large headroom for scheduling under a
tight node budget, so they are the right place to test AZ-GEB.

### Test-set baselines (100-term T3 sample + T4/T5/T7; `checkpoints/geb_probe_<set>_all_<limit>.jsonl`)
| Set | Limit | Terms | egglog default | egglog BackOff-200 |
|---|---|---|---|---|
| T3 | 1,000 | 100 | 87.14 (1,443 e-nodes) | 87.79 (642) |
| T3 | 300 | 100 | 82.46 (678) | 82.47 (452) |
| T4 | 1,000 | 50 | 33.15 (1,011) | 35.06 (615) |
| T5 | 1,000 | 29 | 56.48 (672) | 56.52 (320) |
| T7 | 1,000 | 50 | 70.23 (968) | 70.28 (433) |

BackOff-200 matches or beats egglog default everywhere with 33-56% fewer e-nodes.
Trained AZ-GEB (geb_v7) on T3 at 1,000 nodes: finished only 3 of 30 terms within the 5-minute
limit (mean 276 s per term); on those 3 it equals BackOff-200 (89.02). Run stopped: too slow
on large blocks. The re-implemented MCTS-GEB was dropped (timed out on 48/51 T3 terms).

## Decisions and incidents (dates IST)
- 22 Nov: test assumed built-in rules only; pinned (`tests/test_asm.py`).
- 23 Nov: discovery round 5 run; no gain, no extra arm.
- 23 Nov: egglog OOM in final eval fixed (memory cap + retry), commit c3d36e1.
- 23 Nov: llmft and llmbig skipped to free CPU for AZ-GEB (user decision).
