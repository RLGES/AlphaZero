# Test sets T1-T7: what each contains, and what we learned (24 Nov 2026)

> Terminology: "BackOff-200" = egglog with the back-off scheduler at match limit 200 (ban length 2, up to 12 iterations); "egglog default" = back-off scheduler at match limit 1000, 6 iterations. Earlier notes called BackOff-200 "capped" / "all rules capped".

"Score" = mean % reduction in code cost (higher is better). "@100" = 100 search steps.
Sizes are expression nodes. Buckets: S < 20, M 20-60, L 60-150, XL 151-256.
Where the files are (on the Desktop, inside `~/Desktop/cap/RLGES/AlphaZero/`):
- T1/T2: `benchmarks/asm/corpus/test.jsonl` + `cf_test.jsonl` (T1), `stress_test.jsonl` (T2)
- T3-T7: `benchmarks/asm/testsets/<T>.jsonl` (the terms), `<T>_manifest.json` (every
  per-term measurement), `summary.json` (pass/fail numbers)

## T1 — Held-out (the main test set from run 4)
- **What:** 1,722 real expressions from the Learn-R corpus (compiled programs), never used
  in training. Same set and seeds as night 3 (reproduced exactly: MCTS@100 = 66.1 in both).
- **Shape:** S 778, M 648, L 296; mean 34 nodes, max 149. 48% have a repeated piece;
  29% have a same-operator chain of 6 or more.
- **Results:**
  | Method | Built-in rules | + qwen rules |
  |---|---|---|
  | MCTS @100 | 66.1 | 74.0 |
  | AlphaZero @100 (3 seeds) | 66.3 ± 0.3 | **75.0 ± 0.2** |
  | egglog | 80.5 | 80.8 |
  | egglog then AlphaZero / then MCTS | 80.6 / 80.6 | 80.9 / 80.9 |
  | AlphaZero then egglog | — | 80.7 |
  | egglog on AlphaZero-chosen rules | — | 80.8 |
- **AlphaZero + egglog memory (measured, all 1,722):** same score (80.83 vs 80.79), worst
  case over 2 GB (11 terms needed a retry) -> under 100 MB extra, p95 118 -> 56 MB,
  32% faster, 124 -> 55 rules per term.
- **Conclusions (T1):**
  1. AlphaZero beats MCTS at 100 steps (+0.9 ± 0.2) with about 8x less search work; MCTS
     still leads at 10-25 steps.
  2. The LLM-discovered, Z3-proven rules add **+8.7 [+7.8, +9.7]** to AlphaZero
     (+7.9 to MCTS) but only +0.3 to egglog: discovered rules mainly help step-wise search.
  3. egglog scores highest (80.8) because these expressions are small enough to explore
     almost fully; combining egglog with search after it adds only +0.1.
  4. Letting AlphaZero choose egglog's rules keeps the score and removes the memory
     blow-ups: the clearest "combination is better" result.

## T2 — Stress (longer programs from run 4)
- **What:** 784 harder/longer expressions, not in the corpus.
- **Shape:** S 488, M 233, L 55, XL 8; mean 25 nodes, max 275. 33% with repeated pieces;
  21% with chains of 6+.
- **Results:**
  | Method | Built-in rules | + qwen rules |
  |---|---|---|
  | MCTS @100 | 60.4 | 68.4 |
  | AlphaZero @100 (3 seeds) | 59.4 ± 0.7 | **68.6 ± 0.2** |
  | egglog | 75.6 | 75.7 |
  | egglog then MCTS | 75.7 | 75.8 |
  | AlphaZero then egglog | — | 75.7 |
  | egglog on AlphaZero-chosen rules | — | 75.7 |
- **AlphaZero + egglog memory:** score 75.74 -> 75.70, worst case 577 -> 97 MB extra, 6 retries -> 0, 25% faster.
- **Conclusions (T2):**
  1. AlphaZero ties MCTS (+0.2 ± 0.2) with qwen rules; with built-in rules MCTS is ahead
     by 1.0.
  2. The qwen rules add **+9.2 [+7.8, +10.8]** to AlphaZero, even more than on T1.
  3. egglog again scores highest; the combinations tie it.

## T3 — Large blocks (new)
- **What:** 366 blocks built by joining real T1/T2 expressions (with repeated pieces), sizes
  L 167 / XL 199, mean 142 nodes. Purpose: make egglog struggle.
- **Spec check:** all criteria PASS. egglog failed outright (2 GB cap) on 7.7%, failed or
  built > 3k e-nodes on 15.0% (target 15%); 99.5% have repeated pieces; 100% of best answers
  verified on 200 random inputs.
- **AlphaZero + egglog memory:** score 89.05 -> 88.50, p95 extra RAM 330 -> 28 MB, worst case 888 MB (+12 runs over 2 GB) -> 85 MB, 40% faster. egglog / BackOff-200 / MCTS-GEB baselines still running.

## T4 — Explosion (new)
- **What:** 300 long same-operator chains (10-32 parts) with rules that undo each other
  (x*2 vs x<<1), mean 87 nodes.
- **Spec check:** chains 100% PASS; **egglog stress FAIL** (0% fail or > 5k e-nodes, target
  25%): egglog's back-off scheduler keeps such chains under control.
- **Results:** memory test running.

## T5 — Deep / non-greedy (new)
- **What:** generated deep expressions (depth 4-7) where the greedy method misses the best
  answer.
- **Spec check:** **count FAIL** — only 29 kept (target 200): of 500 candidates, 259 duplicated
  T1/T2 and 82 training terms. 100% of kept terms are non-greedy.
- **AlphaZero + egglog memory (29 terms):** score 56.5 -> 56.1, worst case 285 -> 207 MB,
  0.72 -> 0.53 s per term, no failures.

## T6 — Real whole functions (not built)
Planned in the spec as a feasibility study (multi-output basic blocks from the corpus);
not built: the current term format is single-expression trees BackOff-200 at 256 nodes.

## T7 — Rule-sensitive (new)
- **What:** 249 expressions where the qwen rules change egglog's result (blocks around the
  92 real T1/T2 terms where that happens, plus generated terms).
- **Spec check:** PASS (rule-sensitive on 100%).
- **Results:** memory test running.

## AZ-GEB (AlphaZero steering egglog step by step) — not evaluated on T1-T7
Trained only on training-style examples (never on the test sets); it never learned enough to
be worth testing.
| Version | Setup | Result |
|---|---|---|
| v1 | small terms, enable learned rules | 69.5 vs egglog 70.2 with 5x smaller e-graph; flat during training |
| v2 | enable any rule | invalid (bug: one rule per step was run instead) |
| v3 | large terms, 1,000 nodes | 21.3 (never used the all-rules step) |
| v4-v6 | same | infrastructure failures (memory growth, pool hang, > 2 h per round) |
| v7 | 500 nodes | 70.2 = BackOff-200, above egglog 65.7; copied BackOff-200 every round |
| v8 | 200 nodes | 63.35 = BackOff-200 = egglog; best +0.2 once |
Reason: on these programs the simple BackOff-200 schedule already reaches nearly the best answer
within the memory budget, so step-by-step learning has nothing to gain. The T3 baselines
(egglog / BackOff-200 / MCTS-GEB) will confirm whether any step-wise scheduler can beat BackOff-200.

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

## Overall conclusions so far
- **Works:** AlphaZero beats MCTS efficiently; LLM rules add about +9; AlphaZero choosing
  egglog's rules keeps egglog's score while removing its memory blow-ups and running 32%
  faster.
- **Does not work yet:** AlphaZero steering egglog step by step (AZ-GEB) only matches the
  simple "BackOff-200" schedule; on these programs simple schedules already reach nearly the best
  answer.
- Full numbers and sources: `docs/paper/RESULTS.md`.
