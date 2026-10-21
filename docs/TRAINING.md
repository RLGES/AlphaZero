# Training the AlphaZero engine

The AlphaZero engine (`src/engines/alphazero/`) is PUCT search guided by a
policy/value network. Untrained, it runs with uniform priors and needs no
PyTorch. A trained checkpoint is loaded with `--checkpoint`.

```bash
python -m pip install -e '.[dev,egglog,alphazero,modal]'
./rewrite arithmetic/expand_cancel --engine alphazero --iterations 25
./rewrite arithmetic/expand_cancel --engine alphazero --iterations 25 \
    --checkpoint checkpoints/arith/latest.pt
```

## How training works

The loop follows the PPT training diagram: every round plays self-play games
with the current network, trains on the replay buffer, and publishes a new
checkpoint.

- **Training terms** come from `benchmarks/generators.py` (arithmetic, boolean,
  bitvector). They are random terms with planted redundancy. The hand-written
  benchmarks are never used for training; they are the held-out evaluation.
- **Each self-play move** runs a PUCT search. The root visit counts become the
  policy target.
- **Value target** of a state = the fraction of its cost removed by the best
  cost reached afterwards.
- **Loss** = policy cross-entropy (over legal rules only) + value MSE. AdamW
  weight decay.
- **Checkpoints** store the language and its rule order. Loading one into a
  different language fails loudly.

## Local

```bash
python -m src.engines.alphazero.train --suite arithmetic --rounds 10 --games 64 \
    --sims 32 --steps 200 --workers 6 --out checkpoints/arith
python -m src.engines.alphazero.train ... --resume   # continue from latest.pt
```

Each round writes these files to `--out`:
- `theta_kNNNN.pt` and `latest.pt` (the latter also holds the optimizer state)
- `buffer.pkl`
- one line in `metrics.jsonl`: losses, sample counts, held-out eval cost, timings

## Modal

```bash
modal token new                                   # once per machine
modal run --detach modal_app.py --suite arithmetic --run-name arith_v1
modal run --detach modal_app.py --suite boolean --run-name bool_v1 --rounds 30
modal volume ls rewritezero-checkpoints arith_v1
modal volume get rewritezero-checkpoints arith_v1/latest.pt checkpoints/arith_v1/
```

- **Self-play** fans out over `--chunks` one-CPU containers per round.
- **The driver** runs gradient steps on CPU, which is enough for this network
  size. There is no GPU option, because Modal requires a payment method for GPU
  functions.
- **Resource caps** keep a small budget safe: self-play uses at most 16
  one-CPU/1 GB workers, and the driver runs on 2 CPUs with a 4 GB, 4-hour limit.
- **Disconnects:** `--detach` keeps the run alive if your laptop disconnects.
  The volume is committed after every round, and the driver resumes from
  `latest.pt`.
- **Run names:** reusing a `--run-name` continues that run. Use a new name to
  start fresh.
