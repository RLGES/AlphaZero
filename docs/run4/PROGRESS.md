# RLGES AlphaZero: progress checkpoint

> Claude: read this file first when resuming. Update "RESUME HERE" after every major step.
> - **How night 3 was done, and what was lost:** `RECOVERY_AND_RECIPE.md` (same folder).
> - **Commits:** sumeettummarmatti <sumeettummarmatti@gmail.com>, no Claude trailers.
> - **Pushing:** `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push`. Plain `git push` returns 403.

## ▶ RESUME HERE (last updated 2026-11-24 ~06:20 IST)

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
- 11-23 20:25 IST: AZ-GEB smoke died with the ssh session (not setsid) and was slow: one step could overshoot the node limit (7398 nodes at limit 1000). Fix: back_off(match_limit=200) per step; episode ~8 s under load. Launched training (setsid): checkpoints/geb_v1 (init llm_s0 best.pt, 32 episodes, 16 sims, 1000 nodes, 8 steps, 3 workers, nice 5), log checkpoints/geb_v1/train.log.
- 11-23 20:22 IST: llm_s2 round 43, best 0.632. Full sets, llm_s0: azrules+egglog held-out 80.79 @100 (egglog-all 80.8, so the 60-term +0.2 did not hold), az+egglog stress 75.82, azrules stress 75.70 (egglog 75.7). Seed 1 hybrids running. AZ-GEB round 0 not finished after 40 min (slow on loaded CPU).
- 11-23 20:55 IST: USER: keep everything for writing the paper. Added docs/paper/RESULTS.md (every number so far with its source file) and scripts/paper_figures.py (runs on the server, matplotlib installed via uv; writes docs/paper/figures/fig1_training, fig2_az_vs_mcts_314, fig3_transitions, fig4_final_{heldout,stress} as png+pdf+csv). The hourly push now re-renders figures on the server and commits docs/paper. Keep RESULTS.md updated whenever new results land.
- 11-23 21:20 IST: CRASH fix: AZ-GEB geb_v1 hung 1 h with all workers at 0% CPU (fork after torch init deadlock). Killed those 4 PIDs, switched the pool to spawn (RLIMIT_AS 6 GB), relaunched geb_v1 from scratch at 15:44 UTC; workers at ~85% CPU. llm_s2 round 47, best 0.632 (r35), early stop at r50 if no new best. Seed-1 hybrids on stress running.
- 11-23 22:45 IST: llm_s2 early-stopped r50 (best 0.632); after-eval running since 21:36 IST, then run4_finish.sh triggers the final eval. AZ-GEB v1 flat at 69.49 (egglog 70.17), early-stopped r10; v2 (enable_all, 400 nodes) ~36, stopped by hand. Ablation (40 val terms, 1000 nodes, 8 steps): egglog 70.17/580 nodes, builtin-only 68.45/115, ALL rules with match limit 200 per step 71.69/211, AZ-GEB v1 69.49/118. Next: prune design (start from all rules, learn what to drop) under tighter node budgets. Results in docs/paper/RESULTS.md.
- 11-23 23:13 IST: llm_s2 report @100 AZ 66.8 vs MCTS 66.5 (best-known 90 vs 82). run4_finish watcher missed its trigger (log line comes after the before-eval); appended a marker line by hand at 22:40, it stopped run4.sh/llmft and started final_eval.sh (17:10 UTC). Killed 5 orphaned geb_v2 spawn workers. Final eval builtin done: held-out AZ@100 66.4/66.0/66.5 vs MCTS 66.1 (night 3 builtin 66.5), @10 54.1/52.2/54.6 vs 56.3; stress AZ@100 59.0/58.9/60.3 vs MCTS 60.4; egglog+AZ 80.6 / 75.6. llm arm running. Spec saved docs/paper/TESTSET_SPEC.md. Builder/advocate agents blocked: auto mode cannot review Agent calls because the context-mode plugin hook rewrites prompts; asked user to disable it or let me build inline. v2 enable_all result INVALID (apply() fell through to single-rule branch); fixed + mixed mode added.
- 11-23 23:40 IST: USER: build test sets inline (keep context-mode on). scripts/build_testsets.py (candidates: T3 399 real blocks, T4 300 AC chains len 10-32, T5 500 deep generated, T7 520 rule-sensitive blocks + generated) + scripts/testset_stats.py (all 10 aspects; egglog forked 2 GB cap) + scripts/run_testsets.sh launched 18:05 UTC (2 workers, nice 15), log checkpoints/testsets.log; outputs benchmarks/asm/testsets/<suite>.jsonl + _manifest.json + summary.json. Smoke: ~6.6 s wall per term with 2 workers; ETA ~3.5-4.5 h. Nothing is training now. Explainer session wants paths when final_v4_*.md and final fig4 land.
- 11-24 00:10 IST: Verified every deck number for the explainer session from per-term CSVs (corrections: AZ-then-egglog stress now 3 seeds 75.7; seconds ~9x; builtin MCTS ahead at 10/25/50). AZ qwen final held-out 75.22/74.77/74.98 (mean 74.99); stress s0 68.80, s1/s2 running. Launched scripts/memory_compare.py (peak RSS per egglog child, all rules vs AZ-chosen k=10, 1722 held-out, s0) -> checkpoints/memory_compare_heldout.csv; 10-term smoke: 124 rules 275 MB mean/385 max vs 54.7 rules 260/262, same score.
- 11-24 00:45 IST: memory_compare DONE (1722 held-out): all rules 124 rules, extra RAM mean 63.7 / p95 117.5 / max 931 MB, 11 terms over 2 GB (retried), 3769 matches, 0.71 s; AZ-chosen 54.8 rules, extra 50.2 / 56.1 / 96.4 MB, 0 retries, 2090 matches, 0.49 s; improvement 80.83 vs 80.79. RESULTS.md section 8. Test sets restarted with 5 workers at 18:59 UTC. Cron replaced: 30-min update job d5ef05aa (4,34 * * * *).
- 11-24 00:47 IST: test sets: T3 measured 399/399 (+builtin), T4 289/300, then T5, T7. No errors.
- 11-24 01:17 IST: test sets: T3 399, T4 300, T5 500 measured (+builtin), T7 201/520. T3 egglog-stress 14.8% (<15%), T4 0% (FAIL, egglog never blows up on AC chains). Queued scripts/run_geb_probes.sh (all suites, after build). AZ-GEB suite-style training generator added (geb_train --composite -1); train-side rule-sensitive screening running. 24-term T3 probe still running (slow). Load ~11.
- 11-24 01:55 IST: TEST SUITES BUILT (5010 s): T3 366 (all criteria PASS, egglog stress exactly 15.0%), T4 300 (egglog stress 0% FAIL), T5 only 29 (FAIL: 259 dup of held-out/stress, 82 in training), T7 249 PASS. RESULTS.md section 9. Train-side rule-sensitive pool 265/1500. Started AZ-GEB v3 training (checkpoints/geb_v3: mixed actions, suite-style training terms --composite -1, 1000 nodes, 30 steps, 16 sims, 4 workers) and the baseline probes on all suites (run_geb_probes.sh, 3 workers). 24-term T3 probe still running. TODO: rebuild T5 with deeper/non-overlapping terms; redesign T4 from the held-out terms that blew up egglog.
- 11-24 02:20 IST: geb_v3 round 0 (24 min): AZ 21.3 vs egglog 67.5 vs capped 69.4, AZ e-graphs only 52 nodes: the term network gives the macro action (FOLD index) ~0 prior, so AZ only picked single rules. Stopped v3; added Agent macro_prior (0.5 floor on the macro action). Started geb_v4 (same as v3 + --macro-prior 0.5, 5 workers). Stopped the redundant 24-term probe. Baseline probes (T3 all @1000, 3 workers) running; their az-geb column = untrained AZ without macro prior.
- 11-24 02:47 IST: geb_v4 round 0 still running after 28 min (macro steps make bigger e-graphs, slower). T3 baseline probe running 59 min (3 workers). RAM used 9.1 GiB of 15. No errors.
- 11-24 03:20 IST: geb_v4 workers grew to 1.2-2.6 GB each (egglog e-graphs not freed between episodes), 4.6 GiB free and falling, round 0 unfinished after 59 min. Stopped v4 (PIDs), added max_tasks_per_child=4, started geb_v5 (same settings). Memory back to 12 GiB free. T3 baseline probe still running (1.5 h).
- 11-24 03:47 IST: geb_v5 round 0 running (29 min); largest worker 2.0 GB, 9.9 GiB available. T3 baseline probe at 2 h (writes results only at the end). No errors.
- 11-24 04:20 IST: geb_v5 HUNG (max_tasks_per_child: pool workers exited, main waited at 0% CPU for ~1 h). Stopped it; training now makes a fresh pool per batch of 2x workers jobs. Started geb_v6 (same settings), 5 workers at ~100% CPU. T3 baseline probe at 2.5 h, 3 workers busy.
- 11-24 04:47 IST: geb_v6 round 0 at 30 min, workers busy (0.8-2.1 GB). T3 probe at 3 h. 7.6 GiB available. No errors.
- 11-24 05:22 IST: geb_v6 round 0 at 65 min (not done), T3 probe at 3.5 h. 10 GiB available. No errors. Old ssh launch sessions for v4/v5/v6 closed (exit 255, harmless).
- 11-24 05:47 IST: geb_v6 round 0 at 90 min, still not done; T3 probe at 4 h. Decision: let v6 finish round 0 for a first data point, then restart smaller (fewer episodes/validation/sims, 500 nodes) if rounds stay > 1 h. No errors.
- 11-24 06:20 IST: geb_v6 stopped (round 0 unfinished after 2 h). Started geb_v7 smaller: 16 episodes, 20 validation, 8 sims, 500 nodes, 20 steps, patience 8, 5 workers; batch progress now logged in checkpoints/geb_v7.log. T3 probe at 4.5 h.
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
