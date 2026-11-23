# RLGES AlphaZero: progress checkpoint

> Claude: read this file first when resuming. Update "RESUME HERE" after every major step.
> - **How night 3 was done, and what was lost:** `RECOVERY_AND_RECIPE.md` (same folder).
> - **Commits:** sumeettummarmatti <sumeettummarmatti@gmail.com>, no Claude trailers.
> - **Pushing:** `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push`. Plain `git push` returns 403.

## ▶ RESUME HERE (last updated 2026-11-23 ~19:50 IST)

**Run 4 is in progress on the Vultr server.**
- Server: `ssh <server>`; work dir `/root/rlges`.
- The user's goal: better results than night 3. Push code and results to GitHub. Copy results to the Desktop. Short updates every hour (cron 7bc12980 at :07).

**Where things are**
- Code: `~/Desktop/cap/RLGES/AlphaZero`, branch `run4`, on GitHub as `RLGES/AlphaZero` `run4`.
  - Rebuilt from 906d9ed plus the session logs, then extended.
  - Server copy: `/root/rlges/AlphaZero`; its `.venv` links to `/root/rlges/venv` (CPU torch).
- Learn-R: `~/Desktop/cap/RLGES/Learn-R-git` (branch `eqsat`); on the server at `/root/rlges/Learn-R`, linked as `/root/Projects/RLGES/Learn-R`.
- Old copies, kept as they are:
  - `~/Desktop/cap/RLGES/MCTS-Zero` (iCloud download, 22 Oct)
  - `~/Desktop/Desktop - PES1UG23CS611/cap`

**Pipeline on the server** (all scripts in `scripts/`, all resumable):
1. `run4_data.sh`: rebuilds the hard corpus, control-flow corpus (with dedup) and stress set, using the night-3 commands and seeds. Started 08:27 UTC; log `checkpoints/run4_data.log`.
2. `/root/rlges/launch_run4.sh <data pid>` waits for step 1 and then starts `run4.sh`. Log `checkpoints/run4.log`.
   - `run4_discover.sh` (background): qwen rounds 1–4, each seeded from the terms still stuck. Writes `learned_rules_qwen_all.json`. Log `checkpoints/run4_discover.log`.
   - Meanwhile the builtin arm trains with seeds 0–2 (`asm_v4_builtin_s*`, WORKERS=6).
   - Then the llm arm: all qwen rules, seeds 0–2 (`asm_v4_llm_s*`).
   - Then llmft: each llm seed continued from its best.pt (`--init`, lr 3e-4, 192 simulations, 60 rounds).
   - Then `TAG=v4 ARMS=... final_eval.sh` writes `checkpoints/final_v4_{heldout,stress}.md`.
   - Then llmbig: a 256-hidden, 6-layer network on all rules, seed 0, followed by the final evaluation again (commit b347b8b).
   - Replay buffer left at 50k: it already holds about 20 rounds (2.2–3k samples per round). 200k would keep the weak early games, so it was not adopted.
3. **The LLM runs on the server, not the Mac** (the user's choice, 21 Nov).
   - Docker container `rlges-ollama` (ollama/ollama:0.12.6, `--cpus 4`, <server>:11434 only, volume `rlges-ollama`), model `qwen2.5-coder:7b`.
   - Settings are in `checkpoints/llm.env` (URL, model, timeout 900 s).
   - Measured speed: ~4.5 tokens/s generating, 15 tokens/s on prompts, with the CPU shared. About 2 min per prompt, so rounds 1–4 (750 prompts) take ~1 day. That overlaps with builtin training.
   - After the run, remove it: `docker rm -f rlges-ollama && docker volume rm rlges-ollama`.
   - The Mac is not needed.

**Current state (23 Nov, 10:45 IST)**
- Done: builtin s0–s2 (best 0.552/0.544/0.560); llm s0 (best 0.646, after-eval @100 AZ 67.1 vs MCTS 66.5, best-known 89% vs 82%, ~7.8x fewer transitions).
- Done: llm s1 (early stop r47, best 0.636; @100 AZ 67.3 vs MCTS 66.5, best-known 90% vs 83%).
- Running: llm_s2 training (started 11:25 IST); then, llmft s0–s2, final eval, llmbig (all inside `run4.sh`, do not edit that script while it runs).
- Gap vs night 3: rule lift is +7.3 (run 4, 314 terms) vs +12.7 (night 3). Night-3 qwen12 rule file is lost (overwritten by run-4 round 1–2 files).
- Rule-set check (`scripts/rule_select_eval.py`, MCTS on 300 train-split terms, seed 424242): none 59.4, qwen_all 66.7, all-189-verified pool 66.9 @100. So the 20-per-round filter is not the issue; the factoring rules (x*z+y*z -> (x+y)*z, x*2+x*3 -> x*5) were never found.
- **Discovery round 5** running on the server since 23 Nov 10:36 IST: `scripts/run4_discover5.sh` (Ollama, 200 prompts, seeded from terms stuck with all 80 rules). Logs `checkpoints/run4_discover5.log`, `discover5.log`; output `benchmarks/asm/learned_rules_qwen5.json`.
- Next (all on the SERVER, user's rule): run `rule_select_eval.py` on the server with qwen_all + round 5; if clearly better, add a follow-up script that waits for `run4.sh` to exit, trains an extra arm with the new rules, then reruns `final_eval.sh` including it.
- **User rule (23 Nov): run everything on the server, nothing on the Mac.**


**Each hour:**
- Rsync the server's `checkpoints/` (results, logs, best.pt) to `~/Desktop/cap/RLGES/AlphaZero/checkpoints`.
- Commit the rule files, corpora, tables, logs and best.pt (`git add -f`) and push.

**Report:** a background agent is rebuilding `make_report.py` and writing `~/Desktop/RLGES_Report_v2_draft.pdf`. It contains the night-3 results, the score definition, the architecture, the run-4 plan, and a placeholder table for run-4 results.

## Night-3 reference numbers (to beat)
- Held-out, budget 100: builtin 66.5, llm2 79.2 ± 0.1 (Δ +12.7). Stress: llm2 74.2 (Δ +14.4).
- Budget 10, held-out: llm2 71.0; MCTS leads at budgets 10–25 by 1–2.5 points.

## Log
- 11-21: found `~/Projects/RLGES` missing.
  - Rebuilt the code from GitHub 906d9ed and the session logs; added `--init` continued training and the run-4 scripts.
  - 161 tests pass on the server. Pushed `run4`.
- 11-21 14:22 IST check: hard corpus still filtering (19,400 candidates, 7 workers); run 4 queued; nothing to sync yet.
- 11-21 15:25 IST: hard corpus rebuilt = 10,999 train / 911 test (identical to night 3), pushed. Control-flow corpus building (started 09:48 UTC). Run 4 still queued.
- 11-21 ~16:30 IST: added the egglog baseline for asm (commit 11ca813, deployed; final_eval writes final_v4_*_egglog.md).
  - Pilot on 100 held-out terms with built-in rules: egglog 86.4% vs MCTS@100 71.8%. egglog was never worse on a term; 0.15 s per term; all outputs equivalent on random-input checks.
  - egglog already beats night 3's best AZ (79.2%). Raised with the user: the report must show this honestly; AZ's edge is fewer rewrites. Next idea: RL-guided saturation.
- 11-21 ~16:50 IST: control-flow corpus rebuilt identical to night 3 (cf_train 8,172). Stress set building.
  - Added hybrids to the final evaluation (27414c2, deployed): egglog+MCTS (per rule set) and egglog+AZ (per run).
  - egglog_table shows every method alone and combined, with paired CIs for AZ−egglog, egglog+AZ−egglog and egglog+AZ−egglog+MCTS. Output: checkpoints/final_v4_{heldout,stress}_egglog.md.
- 11-21 16:22 IST (10:51 UTC):
  - Data done at 10:16 UTC: all corpora identical to night 3 (train 10,999 / test 911 / cf 8,172 / 811 / stress 824).
  - Run 4 started 10:16. Discovery round 1 at 26/150 prompts, ~1 min/prompt on Ollama, 20 verified so far.
  - builtin_s0 training: ~7.5 min/round; round 1 score 0.513.
- 11-21 17:15 IST: builtin_s0 at round 8, best 0.539 (night 3 builtin plateaued ~0.55). Discovery round 1 at 79/150 prompts, 58 verified. No errors. Hourly routine script: scratchpad hourly.sh.
- 11-21 18:13 IST: builtin_s0 at round 16, score 0.531, best 0.539 (unchanged since round 8; patience 15 may stop it around round 23). Discovery round 1 at 139/150 prompts, 97 verified. No errors.
- 11-21 19:13 IST: builtin_s0 round 25, score 0.543, new best 0.545. Discovery round 1 done: 105 verified, 20 accepted rules -> learned_rules_qwen.json. Round 2 at 38/200 prompts (19 verified). No errors.
- 11-21 20:13 IST: builtin_s0 round 33, score 0.540, new best 0.550 (matches night-3 builtin ~0.55). Discovery round 2 at 100/200 prompts, 62 verified. No errors.
- 11-21 21:13 IST: builtin_s0 round 41, score 0.545, best 0.552 (round 39, above night-3 builtin ~0.55). Discovery round 2 at 158/200, 77 verified. No errors. Sent run-4 facts to the report agent (capEXPLAINERAGENT).
- 11-21 22:13 IST: builtin_s0 round 50, score 0.544, best 0.552 (round 39; patience 15 -> stops ~round 54, then s1). Discovery round 2 done: 89 verified, total 40 accepted rules (qwen_all). Round 3 at 5/200. No errors.
- 11-21 23:13 IST: builtin_s0 FINISHED (55 rounds, best 0.552). report_after (314 held-out): AZ vs MCTS @10 50.6/54.4%, @25 56.0/58.0, @50 58.3/59.1, @100 59.8/59.9; best-known @100 AZ 83% vs MCTS 81%, AZ ~6.5x fewer transitions. builtin_s1 started (round 0: 0.502). Discovery round 3 at 70/200, 31 verified. No errors.
- 11-22 00:13 IST: builtin_s1 round 8, best 0.536. Discovery round 3 at 129/200, 48 verified. No errors.
- 11-22 01:19 IST: builtin_s1 round 17, best 0.544 (ahead of s0 at the same round). Discovery round 3 at 196/200, 71 verified. No errors. ETA: main results ~Tue 24 Nov 06:00 IST, all incl. llmbig ~Tue 17:00.
- 11-22 02:13 IST: builtin_s1 round 25, best 0.544. Discovery round 3 done (71 verified), 60 accepted rules total; round 4 at 34/200, 11 verified. No errors.
- 11-22 03:13 IST: builtin_s1 stopped early at round 32 (patience), best 0.544 (round 17); its after-eval is running (report_after.txt still empty). Discovery round 4 at 100/200, 42 verified. No errors.
- 11-22 04:13 IST: builtin_s1 report_after: AZ vs MCTS @10 49.4/54.4%, @100 60.1/59.9%, best-known @100 85% vs 81%. builtin_s2 started, round 5 best 0.540. Discovery round 4 at 164/200, 68 verified. No errors.
- 11-22 05:13 IST: DISCOVERY DONE: rounds 1-4 verified 105/89/71/76, 80 accepted rules in learned_rules_qwen_all.json (20 per round). builtin_s2 round 14, best 0.548. Ollama now idle (remove after run). No errors.
- 11-22 06:13 IST: builtin_s2 round 23, best 0.550. Load ~5 (only training now). No errors.
- 11-22 07:13 IST: builtin_s2 round 33, new best 0.5525 (best builtin seed so far). No errors.
- 11-22 08:14 IST: builtin_s2 round 42, new best 0.560 (above night-3 builtin ~0.55). No errors.
- 11-22 09:13 IST: builtin_s2 round 51, score 0.557, best 0.560. No errors.
- 11-22 10:20 IST: builtin_s2 FINISHED (round 55, best 0.560; @100 AZ 59.6% vs MCTS 59.9%, best-known 85% vs 80%). CRASH at 09:59 IST: run4.sh stopped at the llm_s0 test step: test_greedy_misses_what_needs_a_reassociation failed because a qwen rule folds x*2*4 in one step (test isolation, not training). Fixed (test pins built-in rules), 164 passed with both rule sets, relaunched run4.sh at 10:16 IST with setsid nohup; it skipped finished runs/discovery and is on llm_s0 step 4. Old discovery log saved as checkpoints/run4_discover_rounds1-4.log. ~25 min lost.
- 11-22 11:13 IST: llm_s0 before-report (80 qwen rules, untrained AZ vs MCTS, 314 held-out): @10 50.0/62.5%, @25 59.6/65.2, @50 65.2/66.0, @100 66.6/66.5 (best-known 87 vs 83%). With built-in rules MCTS@100 was 59.9%, so the qwen rules add ~6.6 points. llm_s0 training started 11:09 IST (before-eval took 52 min, rules make search slower). No errors.
- 11-22 12:13 IST: llm_s0 round 5, score 0.606, best 0.614 (builtin best was 0.560). ~10 min/round. No errors.
- 11-22 13:13 IST: llm_s0 round 11, best 0.616. No errors.
- 11-22 14:25 IST: llm_s0 round 18, score 0.612, best 0.630. No errors.
- 11-22 15:13 IST: llm_s0 round 23, score 0.624, best 0.630. No errors.
- 11-22 16:13 IST: llm_s0 round 29, score 0.634, best 0.636. No errors.
- 11-22 17:13 IST: llm_s0 round 36, score 0.629, best 0.638. No errors.
- 11-23 11:43 IST: llm_s1 finished: early stop r47, best 0.636; after-eval @10 58.8/62.5, @25 64.0/65.2, @50 66.0/66.0, @100 67.3/66.5 (AZ/MCTS), best-known@100 90% vs 83%. llm_s2 started 11:25 IST. Round 5 discovery prompt 52/200, 13 verified, ETA ~14:30 IST. Load ~11 on 8 cores (training + Ollama), 7.6 GiB RAM avail. No errors.
- 11-23 12:13 IST: llm_s2 round 2, best 0.597 (~16 min/round while Ollama shares the CPU). Round 5 prompt 80/200, 25 verified, ETA ~14:30 IST. No errors.
- 11-23 13:13 IST: llm_s2 round 7, best 0.624. Round 5 prompt 141/200, 47 verified, ETA ~14:15 IST. No errors.
- 11-23 14:13 IST: llm_s2 round 11, best 0.624. Round 5 prompt 197/200, 64 verified, finishing now. No errors.
- 11-23 15:15 IST: round 5 done: 20 loaded + 46 verified (seed_matches mostly 0 = diminishing returns). Rule check @25/@100: qwen_all 65.2/66.7, qwen_all5 (100 rules) 65.4/66.8, pool5 (222) 65.3/67.0: no real gain, so NO extra training arm. Launched scripts/run4_precache.sh (nice 15, 3 workers): MCTS/egglog/egglog+MCTS caches on the 1722 held-out + 784 stress terms for none and qwen_all; reused by final_eval.sh and gives the run-4 rule lift on the same test set as night 3. llm_s2 round 16, best 0.624.
- 11-23 16:13 IST: builtin caches done on 1722 held-out: MCTS @10/25/50/100 = 56.3/63.0/64.6/66.1 (exactly night 3 66.1, so the test set is reproduced), egglog alone 80.5, egglog+MCTS 80.6 (flat across budgets). Stress MCTS @100 60.4 (= night 3). egglog alone beats night-3 llm2 AZ (79.2)! qwen_all caches running. llm_s2 round 22, best 0.631.
- 11-23 17:20 IST: CRASH in precache: egglog with qwen_all hit 11.7 GB on one held-out term, OOM killer (11:22 UTC) broke the pool (training survived). Same would have killed run4 final eval. Fix c3d36e1: final_eval runs egglog in a forked child with RLIMIT_AS 2 GB (EGGLOG_MEM_GB) + 300 s timeout, retries iterations 6/4/2/1, logs to checkpoints/egglog_fallbacks.log; builtin results identical to cache; 164 tests pass on server. Precache relaunched. qwen_all MCTS on 1722 held-out @10/25/50/100 = 66.7/71.8/73.1/74.0 (builtin 66.1 @100 -> lift +7.9; night 3 +12.5). llm_s2 round 27, best 0.631.
- 11-23 18:13 IST: precache DONE (12:23 UTC). Held-out (1722): MCTS@100 none 66.1 / qwen_all 74.0; egglog none 80.5 / qwen_all 80.8; egglog+MCTS 80.6 / 80.9 (flat over budgets). Stress (784): MCTS@100 60.4 / 68.4; egglog 75.6 / 75.7; egglog+MCTS 75.7 / 75.8. 30 egglog memory-cap retries with qwen_all (26 at 4 iterations, 4 at 2), none failed. llm_s2 round 32, best 0.631 (r22) -> early stop at r37 if no new best.
- 11-23 18:40 IST: USER DECISION: do all recs: skip llmbig (watcher scripts/run4_skip_llmbig.sh 3218742 kills run4.sh when it logs asm_v4_llmbig), add AZ->egglog and AZ-chosen-rules->egglog arms (separate), and build AlphaZero acting directly on e-graphs (MCTS-GEB + learning), plan docs/run4/EGRAPH_PLAN.md. Test on 60 held-out terms (llm_s0 best.pt, @100): egglog all 125 rules 86.9 (matches 4100), egglog builtin 86.3, AZ+egglog 86.6, AZrules top-k over ALL rules k=24/40/60: 79.1/79.5/81.4 (bad), builtin + AZ top-k learned + path: k=0 86.7, k=10 87.1 (matches 2643, -35%), k=20 87.1. Launched scripts/run4_azfirst.sh (k=10, 3 workers, nice 15) for llm s0-s2 on both sets. Commit c? pending push.
- 11-23 19:14 IST: llm_s2 round 38, new best 0.632 (r35). AZ-first hybrids: llm_s0 held-out az+egglog done, azrules running. GEB prototype committed (benchmarks/asm/geb.py); 10-term smoke @2000 nodes: egglog 81.5 (0.3 s), MCTS-GEB 62.8 (11 s), AZ-GEB untrained 44.6 (5.5 s). Next: action = builtin iteration + one AZ-chosen learned rule; self-play training loop.
- 11-23 19:50 IST: USER OK: skip llmft too. scripts/run4_finish.sh 3218742 (replaced skip_llmbig watcher) stops run4.sh when it logs asm_v4_llmft, then runs TAG=v4 ARMS="builtin=none llm=qwen_all" final_eval.sh and ablation table. AZ-GEB: guided saturation (kind=enable) + scripts/geb_train.py; 20 rule-sensitive held-out terms: @300 nodes egglog 77.9 / MCTS-GEB 77.4 / AZ-GEB init 75.2; @1000 79.3 / 78.5 / 76.5. Only 79 of 1722 held-out terms are sensitive to the qwen rules under egglog. Training smoke running (slow: >13 min for 4 episodes on a loaded CPU). ETA: AZ-first hybrids Tue 24 midday; run 4 final eval Tue 24 night; AZ-GEB trained results Wed 25 - Thu 26.
- 11-22 18:13 IST: llm_s0 round 42, score 0.637, best 0.638. No errors. Revised ETA: main results ~Tue 24 Nov 21:00 IST, llmbig ~Wed 25 Nov 12:00.
- 11-22 20:13 IST: llm_s0 round 54, score 0.637, new best 0.643. Still improving. No errors.
- 11-22 21:13 IST: llm_s0 round 60, new best 0.644. Still improving. No errors.
- 11-22 22:13 IST: llm_s0 round 67, new best 0.646. Still improving (cap 100 rounds). No errors.
- 11-22 23:13 IST: llm_s0 round 73, score 0.640, best 0.646 (round 67). No errors.
- 11-23 00:13 IST: llm_s0 round 79, score 0.642, best 0.6463 (tiny new best). No errors.
- 11-23 01:13 IST: llm_s0 round 85, score 0.642, best 0.646. No errors.
- 11-23 02:13 IST: llm_s0 early-stopped at round 89 (best 0.646); its after-eval (AZ vs MCTS, step 6) is running since 01:47 IST. No errors.
- 11-23 10:17 IST: llm_s0 after-eval (314 terms) AZ vs MCTS: @10 60.5/62.5, @25 65.0/65.2, @50 66.6/66.0, @100 67.1/66.5; best-known@100 89% vs 82%; AZ ~7.8x fewer transitions. llm_s1 started 02:41 IST (tests 164 pass), round 45, best 0.636. No errors.
- 11-23 10:45 IST: rule-set check (MCTS, 300 train-distribution terms, seed 424242): none 57.4/59.4, qwen_all(80) 65.2/66.7, pool of all 189 verified lowering rules 65.2/66.9 at @25/@100: filter is not the bottleneck, factoring rules are missing. Launched discovery round 5 on server (scripts/run4_discover5.sh, seeded from terms stuck with all 80 rules). User: run EVERYTHING on the server, nothing locally.
