# Training the AlphaZero engine

The AlphaZero engine (`src/engines/alphazero/`) is PUCT search guided by a
policy/value network. Untrained, it runs with uniform priors and needs no
PyTorch. A trained network is loaded with `--checkpoint`; the v1 networks are
in `models/`.

```bash
python -m pip install -e '.[dev,egglog,alphazero,modal]'
./rewrite arithmetic/expand_cancel --engine alphazero --iterations 25
./rewrite arithmetic/expand_cancel --engine alphazero --iterations 25 \
    --checkpoint models/arithmetic_v1.pt
```

## How training works

The loop follows the PPT training diagram: every round plays self-play games
with the current network, trains on the replay buffer, and publishes a new
checkpoint.

- **Training terms** come from `benchmarks/generators.py` (arithmetic, boolean,
  bitvector). They are random terms with planted redundancy; `--depth MIN,MAX`
  controls how hard they are. The hand-written benchmarks are never used for
  training.
- **Each self-play move** runs a PUCT search. The root visit counts become the
  policy target.
- **Value target** of a state = the fraction of its cost removed by the best
  cost reached afterwards.
- **Loss** = policy cross-entropy (over legal rules only) + value MSE. AdamW
  weight decay.
- **Held-out score:** with `--heldout N`, every round also scores the network
  on N generated terms (`--heldout-depth`) from a seed training never uses.
  - `best.pt` is saved whenever that score improves.
  - `--patience P` stops the run after P rounds without improvement.
- **Checkpoints** store the language and its rule order. Loading one into a
  different language fails loudly.

## Local

```bash
python -m src.engines.alphazero.train --suite arithmetic --rounds 20 --games 256 \
    --sims 64 --steps 400 --workers 7 --out checkpoints/arithmetic_v1
python -m src.engines.alphazero.train ... --resume   # continue from latest.pt
scripts/train_local.sh arithmetic boolean bitvector   # one after another, auto-resume
```

Each round writes these files to `--out`:
- `theta_kNNNN.pt` and `latest.pt` (the latter also holds the optimizer state)
- `best.pt` when the score improves
- `buffer.pkl`
- one line in `metrics.jsonl`: losses, sample counts, eval scores, timings

## Overnight on a laptop: the Learn-R pipeline

```bash
scripts/overnight.sh            # whole night; keep the lid open and the charger in
QUICK=1 scripts/overnight.sh    # 2-minute smoke test of every step
tail -f checkpoints/overnight.log
cat checkpoints/overnight_summary.txt   # in the morning
```

The script runs Learn-R → asm suite → AlphaZero end to end. Every step resumes
or skips if it already finished, so rerunning is always safe.

1. **Check** the environment and the Learn-R checkout. It defaults to
   `~/Projects/RLGES/Learn-R`; set `LEARNR_PATH` to use another location.
2. **Fresh run only:** compile 3,300 random Learn-R programs with Learn-R's
   frontend into the asm corpus, and import Learn-R's LLM rules. A rule is
   accepted only if it is still sound once generalised.
3. **Run the tests.** The script stops if any fail.
4. **Before report:** untrained AlphaZero vs MCTS on held-out Learn-R terms.
5. **Train AlphaZero on asm**, with Learn-R's 45 rules as moves. Up to 100
   rounds of 256 games, about 5 minutes per round on a MacBook. It early-stops
   after 12 rounds without a better hard held-out score.
6. **After report** with `best.pt`.
7. **v2 runs** of boolean, bitvector and arithmetic with the remaining time.

Options:
- **Stopping:** Ctrl-C at any point; run the script again to continue.
- **Disk:** the script stops cleanly when less than `MIN_FREE_GB` (default 2) is
  free.
- **Tuning:** `WORKERS` (default 7) and `EXTRA_SUITES` change the run.

## Modal (v2 recipe)

`modal_app.py` defaults to the v2 recipe:

| Setting | v1 (local) | v2 (Modal default) |
|---|---|---|
| Training term depth | 2–5 | 3–7 |
| Moves per game | 24 | 40 |
| Simulations per move | 64 | 128 |
| Network | 128 hidden, 4 layers | 192 hidden, 5 layers |
| Batch | 64 | 128 |
| Rounds | 20 | up to 60 |
| Held-out score / early stop | — | 30 hard terms / stop after 8 flat rounds |

```bash
modal token new                          # once per machine
scripts/modal_v2.sh                      # launch arithmetic, boolean, bitvector; returns at once
ROUNDS=30 scripts/modal_v2.sh boolean    # one suite, capped rounds
scripts/modal_fetch.sh                   # download best.pt, latest.pt, metrics.jsonl
```

- **Watch:** live logs and cost per run are at https://modal.com/apps. Fetched
  results land in `checkpoints/<suite>_v2/`.
- **Single run:** `modal run --detach modal_app.py --suite arithmetic`
  streams that run's log. Any option can be overridden, e.g. `--sims 64 --rounds 20`.
- **Self-play** fans out over `--chunks` (32) one-CPU containers per round, at
  most 32 at a time.
- **The driver** runs gradient steps on 4 CPUs with a 12-hour limit. There is no
  GPU option, because Modal requires a payment method for GPU functions.
- **Disconnects and timeouts:** the volume is committed after every round, and
  running the same command again resumes from `latest.pt`. Reusing a
  `--run-name` continues that run; use a new name to start fresh.

### Cost and time estimate (v2, per suite)

This is measured on a MacBook (one v2 game ≈ 9.5 CPU-seconds) and scaled to
Modal, assuming its CPUs are about 1.5–2× slower. Prices are approximate (about
$0.05 per CPU-hour plus memory); check modal.com/pricing.

| | Per round | 60 rounds (cap) | With early stopping (~25–40 rounds, typical) |
|---|---|---|---|
| Cost | ~$0.08–0.10 | ~$5–6 | ~$2–4 |
| Wall time | ~6–8 min | ~6–8 h | ~3–5 h |

- **All three suites in parallel:** about $8–18 total, finishing in the same
  wall time as one suite.
- **Same recipe on a MacBook:** about 8–9 minutes per round. That's roughly 9
  hours per suite, run one suite at a time.
