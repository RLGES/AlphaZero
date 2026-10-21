# Trained AlphaZero networks (v1)

One policy/value network per suite. These are the final round of a 20-round
local self-play run on 2026-10-21. Each file also holds the optimizer state, so
training can continue from it.

| File | Language | Rules (policy order) |
|---|---|---|
| `arithmetic_v1.pt` | ArithmeticLanguage | simplify, factor, distribute, commute, associate_left, associate_right |
| `boolean_v1.pt` | BooleanLanguage | simplify, demorgan, distribute, factor, commute, associate_left, associate_right |
| `bitvector_v1.pt` | BitvectorLanguage | simplify, demorgan, commute, associate_left, associate_right |

A checkpoint refuses to load into a language with a different rule list.
`*.metrics.jsonl` holds one line per training round: losses, samples, held-out
eval and timings.

## Use

```bash
./rewrite boolean/demorgan --engine alphazero --iterations 25 --checkpoint models/boolean_v1.pt --verbose
python -m src.engines.alphazero.evaluate --suite boolean --checkpoint models/boolean_v1.pt \
    --budgets 10,25,50 --seeds 2 --generated 30
```

## Training setup

```bash
python -m src.engines.alphazero.train --suite <suite> --rounds 20 --games 256 \
    --sims 64 --steps 400 --workers 7 --out checkpoints/<suite>_v1
```

| Setting | Value |
|---|---|
| Rounds | 20 |
| Games per round | 256 (5,120 games per suite) |
| Training examples | about 48k per suite |
| Search | 64 simulations per move, up to 24 moves per game |
| Gradient steps | 400 per round, batch 64 |
| Optimizer | AdamW, lr 1e-3 |
| Network | 128 hidden, 4 layers, 1024 label buckets |
| Training terms | generated, depth 2–5 |
| Hardware | MacBook CPU, 7 self-play workers, about 20 minutes per suite |

## Results

Evaluated on the suite's hand-written benchmarks plus 30 generated terms that
were never used in training (seed 987654321), with 2 search seeds each. The
table shows mean cost reduction: untrained AlphaZero → **trained**, with plain
MCTS in brackets.

| Suite | Budget 10 | Budget 25 | Budget 50 | Rewrites applied @25, AZ vs MCTS |
|---|---|---|---|---|
| arithmetic | 43.3% → **45.8%** (45.4%) | 45.7% → **50.1%** (50.1%) | 48.1% → **51.0%** (50.8%) | 27 vs 178 |
| boolean | 74.1% → **78.2%** (66.2%) | 78.4% → **79.5%** (75.9%) | 79.5% → **81.2%** (77.9%) | 21 vs 179 |
| bitvector | 38.5% → **48.1%** (50.1%) | 46.6% → **50.8%** (52.2%) | 49.9% → **51.9%** (52.7%) | 25 vs 106 |

- **Training improves every suite at every budget.**
- **Trained AlphaZero vs MCTS:** it beats MCTS on boolean, ties on arithmetic,
  and is slightly behind on bitvector, while applying 4–8× fewer rewrites.
- **Wall-clock time:** AlphaZero is slower, because each network call costs
  more than a random rollout step.
- **Losses** flattened after about 10 rounds, and the training terms are easy.
  The v2 run uses harder terms, deeper search and early stopping on a hard
  held-out set; see `docs/TRAINING.md`.
