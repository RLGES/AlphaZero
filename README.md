# RLGES AlphaZero: learned search for term rewriting

Part of the **RLGES** capstone (Reinforcement Learning Guided Equality
Saturation). This repository optimises terms with rewrite rules using three
engines that share one language-agnostic API:

| Engine | What it does | Needs |
| --- | --- | --- |
| **AlphaZero** | PUCT tree search guided by a policy/value network trained by self-play | PyTorch for trained networks |
| **MCTS** | Hand-written UCT Monte Carlo tree search with random rollouts | nothing |
| **egglog** | Bounded equality saturation, for comparison (arithmetic and Boolean) | `egglog` |

The **asm** suite connects the engines to **Learn-R**: its terms come from programs
compiled by Learn-R's frontend, and its 45 actions are Learn-R's built-in rewrite
rules (plus any LLM-learned rules that stay sound when generalised).
`scripts/overnight.sh` runs the whole Learn-R → AlphaZero pipeline on a laptop.

The term-rewriting framework, its benchmark suites and the MCTS and egglog
engines come from **RewriteZero / MCTS-Zero by Sujay Karpur**. The AlphaZero
engine, training pipeline, term generators and trained networks are built on
top of it.

## Quick start

Python 3.12 or newer is required (egglog 14 needs it). Run these commands from
the repository root (on Windows, use WSL):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,egglog,alphazero]'

./rewrite list                                              # all 27 benchmarks
./rewrite arithmetic/factor_once --verbose                  # MCTS (default)
./rewrite arithmetic/factor_once --compare-egglog
./rewrite boolean/demorgan --engine alphazero --iterations 25 \
    --checkpoint models/boolean_v1.pt --verbose             # trained AlphaZero
./rewrite suite arithmetic --engine alphazero --iterations 50
python -m pytest -q
ruff check .
```

Extras:
- `.` (core): MCTS and the untrained AlphaZero engine, no dependencies.
- `[dev]`: tests and lint.
- `[egglog]`: the comparison engine.
- `[alphazero]`: PyTorch, needed for trained networks and training.
- `[modal]`: cloud training.

After installation `rewrite` is also a console command. Without installation,
`python rewrite ...` and `python -m src.api.cli ...` work from the repository
root.

Single runs report initial and best costs, percentage improvement, elapsed time,
iterations, unique states, and the first iteration that found the best result.
Suite and aggregate runs print a table. `--verbose` adds terms, the rewrite
trace, and engine statistics. Missing egglog installations and unsupported
suites produce an explanatory message while the main engine still runs.

## AlphaZero engine

`src/engines/alphazero/` implements the training and inference loops from the
RLGES design.

- **Game:** the state is a term, a move applies one rewrite rule to all its
  matches, and the score is the cost reduction.
- **Network** (`network.py`): a tree message-passing policy/value network in
  plain PyTorch. The policy scores the language's fixed rule list. The value
  predicts the fraction of the term's cost that can still be removed.
- **Search** (`search.py`): PUCT with network priors and leaf values instead of
  random rollouts. It shares the MCTS engine's transposition and cycle
  handling. Without a checkpoint it runs with uniform priors and needs no
  PyTorch.
- **Self-play and training** (`selfplay.py`, `training.py`, `train.py`):
  - every round plays games on generated terms and records search policies and
    value targets
  - it trains on a replay buffer and saves resumable checkpoints
  - optionally, it scores the network on hard held-out terms each round, keeps
    `best.pt`, and stops early when the score stops improving
- **Cloud training** (`modal_app.py`, `scripts/modal_v2.sh`): self-play fans out
  over Modal CPU containers.
- **Evaluation** (`evaluate.py`): compares MCTS, AlphaZero and egglog on
  held-out terms, reporting rewrite applications and wall time as well as
  iterations.

```bash
python -m src.engines.alphazero.train --suite arithmetic --rounds 20 --workers 7 \
    --out checkpoints/arithmetic_v1                        # train locally
scripts/modal_v2.sh                                        # train v2 on Modal
python -m src.engines.alphazero.evaluate --suite boolean \
    --checkpoint models/boolean_v1.pt --generated 30       # compare engines
```

Trained **v1** networks for arithmetic, boolean and bitvector are in `models/`.
On held-out terms, training improves the engine on every suite at every budget.

| Suite | Trained AlphaZero vs plain MCTS | Rewrites applied @25 (AZ vs MCTS) |
| --- | --- | --- |
| boolean | better at every budget | 21 vs 179 |
| arithmetic | equal quality | 27 vs 178 |
| bitvector | 1–2 points behind | 25 vs 106 |

AlphaZero applies 4–8× fewer rewrites than MCTS for that quality, but it is
slower in wall time. Full numbers are in [`models/README.md`](models/README.md).
How training works, the v2 recipe, and Modal costs are in
[`docs/TRAINING.md`](docs/TRAINING.md).

## Python API

The public package is deliberately named `src` to match the flat source layout,
without an extra `src/rewritezero/` package layer:

```python
from src import AlphaZeroConfig, MCTSConfig, rewrite, rewrite_detailed
from benchmarks.arithmetic import Add, Mul, Var, ArithmeticLanguage

x, y, z = Var("x"), Var("y"), Var("z")
term = Add(Mul(x, y), Mul(x, z))
language = ArithmeticLanguage()

best = rewrite(language, term, MCTSConfig(iterations=100, seed=0))  # x * (y + z)
result = rewrite_detailed(
    language, term, AlphaZeroConfig(iterations=25, checkpoint="models/arithmetic_v1.pt")
)
print(result.engine, result.initial_cost, result.final_cost)  # AlphaZero 7.0 5.0
print(result.trace, result.statistics)
```

`rewrite(language, term, engine_config) -> term` returns the cheapest term
encountered, including the input. Omitting the config selects MCTS.
`rewrite_detailed` returns a `RewriteResult` with the input, final/best term,
costs, wall time, engine name, trace, and statistics.

`EngineConfig.create_engine()` dispatches the engine; implementing this small
method and `Engine.run()` is enough to add another search strategy.
`EgglogConfig(iterations=6, match_limit=1000)` selects the comparison backend.

## Layout

```text
rewrite                        executable CLI
pyproject.toml                 packaging, extras, and test/lint configuration
modal_app.py                   Modal cloud training (CPU)
src/
  api/                         public API and benchmark CLI
  language/                    Term, RewriteRule, Language
  core/                        Engine, EngineConfig, RewriteResult, TraceStep
  engines/
    mcts/                      configuration and hand-written search
    egglog/                    optional engine and local adapter protocol
    alphazero/                 features, network, PUCT search, self-play, training, evaluation
benchmarks/
  common.py                    benchmark metadata, tree batch-rewrite helper, graph features
  registry.py                  explicit suite discovery and lookup
  generators.py                training-term generators (arithmetic, boolean, bitvector)
  arithmetic/ boolean/ lambda/ bitvector/ linear_algebra/ regex/
models/                        trained v1 networks, training metrics and results
scripts/                       local training queue, Modal launch and fetch
docs/TRAINING.md               training guide, v2 recipe, Modal costs
tests/                         engine, semantics, registry, CLI, egglog and AlphaZero tests
```

Each suite contains its own immutable terms, rules, costs, and benchmark definitions.
The shared tree traversal helper belongs to `benchmarks/`; the core and engines
never inspect an AST, node operator, child list, binder, or matrix dimension.

## Search

One search action is whatever the language returns from `applicable(term)`.
In the supplied tree suites it selects a rule, and the language applies that rule
to a deterministic batch of non-overlapping matches:

1. Inspect the original root first.
2. If it changes, replace it and skip its descendants for this action.
3. Otherwise recurse left-to-right and rewrite disjoint siblings.
4. Do not revisit newly created subterms.

Thus, removing the outer zero in `0 + (x + 0)` takes one action and leaves
`x + 0`; applying the same rule again produces `x`.
The benchmark helper excludes no-ops and expansions beyond 256 nodes by default;
`Language` itself imposes no tree or size requirements.

MCTS performs UCT selection, one-edge expansion, random rollout, and
backpropagation. Its reward is the best cost along the sampled path, normalized
by the input cost. It saves the best term immediately, even if later rollout
steps increase the cost. Transpositions share states and cached outgoing
transitions; action visit counts and rewards live on edges. Cycles are detected
against the current path, allowing the same state to be explored by other paths.
AlphaZero keeps this structure but replaces UCT and rollouts with PUCT and one
network evaluation per new leaf.

Default MCTS configuration:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `iterations` | 1000 | Maximum search iterations; zero returns the input |
| `rollout_depth` | 12 | Maximum random transitions after expansion |
| `exploration_constant` | sqrt(2) | UCT exploration weight |
| `seed` | 0 | Local RNG seed; use `None` for unseeded runs |
| `max_tree_depth` | 64 | Maximum selection/expansion depth per iteration |
| `max_states` | 10000 | Bound on cached unique states |
| `time_limit` | None | Optional elapsed-time limit in seconds |

`AlphaZeroConfig` has the same budget fields, plus:
- `c_puct` (1.5)
- `checkpoint` or an in-memory `network`
- `dirichlet_alpha` and `dirichlet_epsilon` (root exploration noise, used in
  self-play)
- `device`

Statistics include iterations, states explored, first best iteration,
transpositions, cycles, transition evaluations, rollout steps (MCTS) or network
evaluations (AlphaZero), and stop reason. Time limits are checked between
transitions; a language action already executing runs to completion. With
iteration/state budgets, deterministic languages, and a fixed seed, terms,
traces, and search statistics are reproducible. Wall time is not deterministic,
and elapsed-time budgets can change the search path.

This is a bounded research search, with no claim of globally optimal results.

## Languages and benchmarks

There are 27 hand-constructed benchmarks. Difficulty labels describe expected
optimization/search difficulty and never change engine behavior. AlphaZero
training uses generated terms only; these benchmarks stay held out.

| Suite | Semantics and cost | Examples | egglog |
| --- | --- | --- | --- |
| arithmetic | Exact, unbounded commutative integers; node count | `factor_once`, `constant_chain`, `expand_cancel` | Yes |
| boolean | Propositional Boolean algebra; node count | `demorgan`, `absorption`, `complement_factor` | Yes |
| lambda | Pure untyped lambda calculus, capture-avoiding beta and guarded eta; node count | `nested_beta`, `capture_avoidance`, `duplicate_then_reduce` | No |
| bitvector | Unsigned fixed-width words, modular addition and bitwise operations; node count | `wraparound`, `xor_cancel`, `demorgan_mask` | No |
| linear_algebra | Shape-checked exact matrices; estimated scalar operations plus leaf cost | `chain_order`, `common_left`, `transpose_chain` | No |
| regex | Classical regular languages with set union and Kleene star; node count | `common_prefix`, `nested_star`, `expand_then_absorb` | No |

The arithmetic cancellation, lambda duplication, and regex absorption examples
exercise temporarily larger terms. Matrix chain reassociation demonstrates a
cost function that depends on dimensions rather than AST size.

Arithmetic uses integer identities, not floating-point reassociation. Matrix
multiplication preserves operand order. Regex terms have no captures or ordered
alternation. Lambda reductions preserve beta/eta equivalence; MCTS minimizes
syntax cost rather than guaranteeing a normal form. Named lambda binders are
preserved in fingerprints, so alpha-equivalent terms can occupy separate states.
The `lambda` directory is loaded through `importlib` because `lambda` is a Python
keyword; for direct access use `import_module("benchmarks.lambda.term")`.

## Egglog comparisons

The backend uses the Python [egglog package](https://egglog-python.readthedocs.io/)
with suite-local adapters that encode terms, define algebraic equalities, extract
using the suite's cost model, and decode back to the original IR. It does not
enumerate MCTS states to simulate equality saturation.

The arithmetic adapter uses egglog's `BigInt` primitive so constant folding agrees
with Python's unbounded integers. Extraction assigns cost only to language IR
nodes, ignoring egglog primitive payload overhead. Both engines report the
original language cost and keep the input if no cheaper candidate is found.

Egglog runs a bounded schedule, with a backoff scheduler limiting large rule
match batches. Six rounds need not reach full saturation. MCTS and egglog have
different search policies and budgets even when they encode the same identities.
Egglog has no sequential rewrite trace.

Elapsed time uses `time.perf_counter()` and includes per-run setup, search, and
extraction. The first egglog run also includes lazy package/adapter imports;
`statistics["setup_time"]` exposes setup overhead. Compare repeated runs when
studying warm backend timings.

## Extending the framework

Implement a `Language[YourTerm, YourAction]` with:

- `applicable(term)`: a stable sequence of available actions.
- `apply(term, action)`: a deterministic transition without mutating the input.
- `cost(term)`: a finite, nonnegative number.
- `fingerprint(term)`: a stable, hashable identity that distinguishes search states.
- `format(term)`: optional readable output.

Your term may be a tree, a graph, a list, or another IR; it need not be hashable.
Fingerprints must be collision-free for the states you want search to distinguish.
Add benchmark definitions and register the suite name in `benchmarks/registry.py`.
For optional egglog support, provide a suite-local `egglog_adapter()` factory
implementing `src.engines.egglog.adapter.EgglogAdapter`.

For AlphaZero, the language also provides:
- `features(term) -> GraphFeatures`: node labels plus (parent, child, position)
  edges.
- `action_vocabulary()`: a fixed, ordered superset of every `applicable()`
  result.

Tree languages get both from `TreeLanguage`; override `node_label` when a node's
operator alone isn't enough. To train on a new suite, add a
`benchmarks/<suite>/generator.py` with `random_term(rng, depth)` and register it
in `benchmarks/generators.py`.

## Validation

`pytest` covers engine dispatch, non-tree and unhashable terms, seeded
reproducibility, cycles, transpositions, temporary cost increases, retained
intermediate states, budgets, deterministic batch matching, all suite lookups,
all benchmark smoke runs, CLI handling, and supported egglog comparisons.
Semantic checks exercise arithmetic values, Boolean truth tables, bitvector
wraparound, lambda capture avoidance, matrix values/shapes, and regex membership.
The AlphaZero tests cover features, the network, PUCT search, self-play targets,
training, checkpoints, early stopping, term generators and the Modal app.

Egglog-, PyTorch- and Modal-dependent tests skip cleanly when the optional
package is absent. Run `python -m pytest -q` and `ruff check .` locally; there
is no CI workflow for now.
