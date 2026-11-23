# Test-set specification for e-graph search (AlphaZero-GEB) — agreed 23 Nov 2026

Why: on the existing test sets egglog saturates almost every term (only 79 of 1,722
held-out terms change when the qwen rules are added), so a learned e-graph builder
has nothing to win. A prototype of joined held-out terms (4 per block) made egglog
fail under a 2 GB cap on 3-4 of 20 terms at 3k-10k e-nodes, while "all rules,
match limit 200 per step" stayed at 87.1 with ~640 e-nodes; at a 300-node limit both
lost ~5.5 points. Learned scheduling can only show value on such terms.

All sets are built and measured on the server. Every acceptance criterion below is
measurable from the set's manifest and a re-run of `scripts/testset_stats.py`.

## Aspects every suite must report (per term, stored in the manifest)
1. **Size**: node count; bucket S (<20), M (20-60), L (60-150), XL (150-500).
2. **Shape**: depth, max width, number of repeated subtrees of >= 3 nodes (sharing).
3. **Operator mix**: counts per op class: arithmetic (add/sub/mul), bitwise
   (and/or/xor/not/shl/shr), expensive (div/mod), memory (load), control (phi);
   domain label pure-arith / pure-bitwise / mixed.
4. **Constants**: constant count, powers of two, masks (2^k-1), constants that fold.
5. **E-graph stress**: longest chain under one commutative-associative op (add, mul,
   and, or, xor); egglog default schedule at 10k e-nodes under a 2 GB cap: final
   e-nodes, seconds, failed/capped flag.
6. **Difficulty**: greedy cost vs best-known cost (non-greedy flag via
   `benchmarks/asm/hardness.py`); best-known = min over egglog (capped), all-rules
   capped schedule, MCTS@100.
7. **Rule sensitivity**: egglog best cost with built-in rules vs with qwen rules.
8. **Provenance**: real (Learn-R corpus / held-out), synthetic, adversarial, external.
9. **Correctness**: every term parses, evaluates, is within the size cap; any
   rewritten output used for best-known is checked equal on 200 random inputs
   (32-bit semantics incl. div/mod by zero, shifts >= 32, overflow).
10. **Separation**: no term (up to variable renaming, by structural fingerprint)
    appears in training generators' seeds (< 100,000), in another suite, or twice.

## Suites and acceptance criteria
| Suite | Content | Acceptance (all must hold) |
|---|---|---|
| T3 Large blocks | held-out / test-split terms joined into blocks, with shared subterms | >= 300 terms; >= 100 in L and >= 100 in XL; >= 30% with sharing; egglog default fails or exceeds 3k e-nodes on >= 15% |
| T4 Explosion | long AC chains, rules that undo each other (x*2 <-> x<<1), dense matches | >= 200 terms; >= 60% with an AC chain of length >= 6; egglog default fails or exceeds 5k e-nodes on >= 25% |
| T5 Deep / non-greedy | many rewrite steps, greedy misses the optimum | >= 200 terms; non-greedy (greedy > best-known) on >= 60% |
| T6 Real functions | whole functions / multi-output basic blocks from the Learn-R corpus | feasibility report; build if the term representation allows; else documented why not |
| T7 Rule-sensitive | terms where the qwen rules change egglog's best cost | >= 200 terms; qwen-rules egglog cost < built-in egglog cost on 100% |
| T8 External | egg Math / Prop benchmarks used by MCTS-GEB | feasibility report (needs those rule sets and languages); build only if feasible without breaking anything |

Every suite: per-bucket counts, operator-mix table, provenance, separation check,
correctness check, build time and measurement time recorded in its manifest.

## Budgets and metrics for experiments on these suites
- E-node limits 100 / 300 / 1k / 3k / 10k; 2 GB memory cap; failures counted.
- Per term: initial cost, final cost, improvement, gap to best-known, e-nodes,
  seconds, failed/capped. Results reported per size bucket; >= 3 seeds for learned
  methods; paired bootstrap 95% CIs.

## Constraints while building
- Run everything on the server (`/root/rlges/AlphaZero`). The final evaluation of
  run 4 is running there: do not modify `benchmarks/asm/final_eval.py`,
  `egglog_adapter.py`, `language.py` behaviour, or anything it imports. Raising the
  256-node cap must be opt-in (env var, default unchanged) or confined to new files.
- New code in new files: `benchmarks/asm/testsets/` (data, manifests),
  `scripts/build_testsets.py`, `scripts/testset_stats.py`.
- Low priority (`nice -n 15`), at most 2 workers, while the final evaluation runs.
