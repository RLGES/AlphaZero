# Run 4 extension: AlphaZero and e-graphs

Decided 23 Nov 2026. All runs on the server. Run 4's baseline plan continues
unchanged except llmft and llmbig, which are skipped to free the CPU for AZ-GEB (`scripts/run4_finish.sh` stops run4.sh at the llmft step and runs the final evaluation for builtin and llm).

## Why
On the 1,722 held-out terms (qwen_all rules): egglog 80.8, egglog+MCTS 80.9,
MCTS@100 74.0, AlphaZero@100 ~74-75 (estimate from the 314-term reports).
Running search after egglog adds ~0.1, so "egglog then search" cannot show that
the combination is better. With many rules egglog's e-graph also explodes
(11.7 GB on one held-out term; 30 terms needed the memory cap).

## Arms (each kept separately in the tables)
1. **egglog → AZ** (existing `egglog+az`): AZ searches from egglog's extracted term.
2. **AZ → egglog** (`az+egglog`): AZ rewrites first at budget B, egglog saturates
   from AZ's result with all rules.
3. **AZ-chosen rules → egglog** (`azrules+egglog`): egglog saturates the original
   term using only the rules AZ picks for it: the top-k rules by the network's
   prior at the root, plus every rule on AZ's best rewrite path. Goal: same or
   better quality with a far smaller e-graph, which lets egglog use the 222-rule pool.
4. **AlphaZero-GEB** (AZ acting directly on e-graphs; trained): MCTS-GEB
   (He, Singh, Yoneki, EuroMLSys '23) with learning.
   - State: an e-graph (egglog `push`/`pop` instead of MCTS-GEB's action replay).
   - Action: apply one rule for one egglog iteration (the same action space as
     MCTS-GEB and as our AZ policy head, so the trained term network initialises it).
   - Reward: relative cost reduction of the extracted term; episode ends at the
     e-node limit or a step limit.
   - Network: existing GNN on the currently extracted term + per-rule match counts
     + e-graph size -> prior over rules and a value. PUCT, no random rollouts.
   - Baselines on the same e-node budgets: egglog's default schedule, MCTS-GEB
     (random rollouts, UCT), AZ-GEB untrained (priors from the term network), AZ-GEB trained.

## Ideas extending MCTS-GEB to AlphaZero (for the paper)
- Value network replaces random rollouts. MCTS-GEB spends 400-2000 s per expression
  on rollouts; one network call per leaf should cut this by orders of magnitude.
- Policy prior over rules (PUCT) replaces uniform expansion; it prunes rules that
  only grow the e-graph. MCTS-GEB's own future work asks for action pruning.
- Memory as the budget: report quality versus e-node limit (1k, 2k, 5k, 10k). A
  learned schedule should reach egglog's quality with a fraction of the nodes.
- Generalisation: train on generated terms, test on the held-out Learn-R corpus and
  the stress set; MCTS-GEB plans per expression from scratch.
- Large rule sets: LLM-discovered rules make default saturation blow up; a learned
  scheduler can use all 222 verified rules where egglog needs a memory cap.
- Warm start from the term-level AlphaZero policy (same action vocabulary).
- Tree reuse between steps and push/pop e-graph snapshots (MCTS-GEB future work).

## Order
1. Skip llmbig (watcher). 2. Arms 2-3 in `final_eval.py`, small test, then full sets
   for llm s0-s2. 3. AZ-GEB environment, MCTS-GEB baseline, untrained AZ-GEB, small test.
4. Train AZ-GEB once run 4's training ends; evaluate on node budgets.
