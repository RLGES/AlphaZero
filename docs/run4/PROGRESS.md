# RLGES AlphaZero: progress checkpoint

> Claude: read this file first when resuming. Update "RESUME HERE" after every major step.
> - **How night 3 was done, and what was lost:** `RECOVERY_AND_RECIPE.md` (same folder).
> - **Commits:** sumeettummarmatti <sumeettummarmatti@gmail.com>, no Claude trailers.
> - **Pushing:** `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push`. Plain `git push` returns 403.

## ▶ RESUME HERE (last updated 2026-11-22 ~06:13 IST)

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
