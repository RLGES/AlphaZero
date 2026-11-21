# RLGES: how night 3 was done, and the rebuild plan

Written 2026-11-21 after `~/Projects/RLGES`, which held all work from 22 to 26 Oct, was found missing.
Where the record survives:
- The final report: `~/Desktop/RLGES_Night3_Report.pdf` (26 Oct 21:16, 3 seeds for every arm).
- The session logs in `~/.claude/projects/-Users-sumeet-Desktop-cap-RLGES/`.

## What survives

| Item | Where |
|---|---|
| Code up to the morning of 22 Oct (night 1, morning v2 runs) | `~/Desktop/cap/RLGES` (iCloud copy, includes Learn-R, RL-encoder and capstone-docs); GitHub `RLGES/AlphaZero` `learnr-bridge` @ 906d9ed |
| Old local partial copy (night-1 checkpoints) | `~/Desktop/Desktop - PES1UG23CS611/cap/RLGES` |
| Files written whole, recoverable word for word from the session logs | `scripts/night3.sh`, `scripts/final_eval.sh`, `scripts/ablation_table.py`, `benchmarks/asm/final_eval.py`, `benchmarks/asm/discover.py`, `benchmarks/asm/hardness.py`, `plans/report/make_report.py`, `PROGRESS.md` |
| Edited files, partly recoverable from the logs | `learnr_bridge.py` (`--statements`, `--max-nodes`), `overnight.sh` (SEED, HELDOUT, PATIENCE, GENERATED overrides), `tests/test_asm.py`, Learn-R `rule_parser.py` (accepts Markdown replies) |
| LOST: must be regenerated | `learned_rules_qwen.json` (round 1), `learned_rules_qwen2.json`, `learned_rules_qwen12.json` (40 rules), hard and control-flow corpora, `stress_test.jsonl`, the 9 `asm_v3_*` trained networks, the eval CSVs |

## How it was done last time

**Cost model (asm suite).**
- mul 3, div/mod 10, load 4, every other op 1.
- 32-bit semantics: div/mod by 0 = 0; shift counts are taken & 31.

**Built-in rules.** Learn-R's 44 rules plus constant folding ("45 built-in").

**Corpora**, built with `learnr_bridge` from Learn-R-compiled random programs:
- *Hard corpus (commit e440fd5):* 10,999 train / 911 test terms; 86% (train) and 100% (test) hard. It plants commuted and nested redundancy.
  - Hardness filter: keep a term only if quick search falls short of strong search.
- *Control-flow corpus (76a442d):* if/while programs split into basic blocks. 8,172 train / 811 test terms, after removing 18 terms that appeared in both train and test.
- *Totals:* 19,171 train and 1,722 held-out terms (test + cf_test, deduplicated).
- *Stress set (9179943):* seed 2026, 50 programs, 15–28 statements, max-nodes 300, cf share 0.25. Gives 824 terms, or 784 after removing any term that appears in a corpus.

**Qwen discovery** (`discover.py`; qwen2.5-coder-7b-instruct in LM Studio at localhost:1234):
- *Pipeline:* duplicate filter, then random testing, then a Z3 proof over 32-bit bit-vectors (`smt.py`). `rank_rules` loads only rules that lower cost.
- *Round 1:* 604 proposals, 70 proved, 20 loaded. Mostly div/mod/shift rules, so no gain on Learn-R code.
- *Round 2:* seeded from terms still stuck with round-1 rules, with the round-1 rules passed as `--known` (counted as duplicates). 973 proposals, 66 proved, 20 loaded.
  - Merged with round 1 into `learned_rules_qwen12.json` (40 rules; round-2 names prefixed `r2_`).
  - Examples: (x+y)-(y+z) -> x-z; x*z+y*z -> (x+y)*z; x*2+x*3 -> x*5.

**Model and training** (`overnight.sh` via `night3.sh`):
- *Search and network:* AlphaZero with PUCT. GNN with 5 message-passing layers, 192 hidden, child-position embeddings, hashed labels, readout = root vector + mean pool.
- *Self-play:* 256 games per round, 96 simulations per move, 400 gradient steps per round.
- *Checkpoint selection:* best.pt chosen on 300 generated held-out terms (TEST_SEED split: ~60% corpus test terms, ~40% synthetic div/mod/shift/load). Score = mean relative improvement at 25 iterations.
- *Run settings:* `HELDOUT=300 PATIENCE=15 GENERATED=150 EXTRA_SUITES=""`. Seeds 0, 1, 2.
- *Arms:*
  - builtin: `RLGES_ASM_LEARNED=none`
  - llm: `RLGES_ASM_LEARNED=learned_rules_qwen.json`
  - llm2: `RLGES_ASM_LEARNED=learned_rules_qwen12.json`
- *Run directories:* `checkpoints/asm_v3_<arm>_s<seed>`. Training plateaued after about 35–45 rounds.

**Final evaluation** (`final_eval.py`, `final_eval.sh`, 7 workers):
- *Terms:* every best.pt scored on all 1,722 held-out terms and the 784 stress terms.
- *Budgets:* 10, 25, 50 and 100.
- *MCTS baseline:* run once per rule set and cached.
- *Statistics:* paired per-term bootstrap 95% CI; mean ± std over the 3 seeds.

**Final results (26 Oct):**

| Set | Rule set | AZ at budget 10 | AZ at budget 100 | MCTS at budget 100 | Δ vs builtin at budget 100 |
|---|---|---|---|---|---|
| Held-out | builtin | 54.7 | 66.5 | 66.1 | — |
| Held-out | llm | 54.4 | 66.7 | 66.1 | ≈ 0 |
| Held-out | llm2 | 71.0 | 79.2 ± 0.1 | 78.6 | +12.7 [+11.7, +13.8] |
| Stress | builtin | 49.0 | 59.8 | 60.4 | — |
| Stress | llm2 | 66.3 | 74.2 | 74.1 | +14.4 |

- Δ at budget 10 for llm2: +16.2 on held-out, +17.2 on stress.
- AlphaZero uses about 6x fewer rewrites than MCTS (93 vs 556 at budget 100).

## Server
- CPU-only cloud server (8 vCPU, 15 GiB RAM); details kept off the repo.
