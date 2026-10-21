# RewriteZero v0

A small, language-agnostic framework for optimizing terms with rewrite systems.
The main engine is a hand-written Monte Carlo tree search (MCTS) implementation.
An optional Python egglog backend compares search with bounded equality saturation
for the arithmetic and Boolean suites.

## Quick start

Python 3.11 or newer is required. Run these commands from the repository root
(on Windows, use your WSL terminal):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,egglog]'

./rewrite arithmetic/factor_once
./rewrite arithmetic/factor_once --compare-egglog --verbose
./rewrite boolean/demorgan
./rewrite lambda/nested_beta
./rewrite suite arithmetic --iterations 100 --seed 7
./rewrite all --iterations 100 --compare-egglog
./rewrite list
python -m pytest -q
ruff check .
```

MCTS has no runtime dependencies. Install with `pip install -e .` for the core,
`pip install -e '.[dev]'` for tests, or `pip install -e '.[egglog]'` for comparison.
After installation, `rewrite` is also available as a console command.
Without installation, `python rewrite ...` and `python -m src.api.cli ...` work
from the repository root.

Single runs report initial and best costs, percentage improvement, elapsed time,
iterations, unique states, and the first iteration that found the best result.
Suite and aggregate runs print a table. `--verbose` adds terms, the MCTS rewrite
trace, and engine statistics. Missing egglog installations and unsupported suites
produce an explanatory comparison message while still running MCTS.

## Python API

The public package is deliberately named `src` to match the requested flat
source layout, without an extra `src/rewritezero/` package layer:

```python
from src import MCTSConfig, rewrite, rewrite_detailed
from benchmarks.arithmetic import Add, Mul, Var, ArithmeticLanguage

x, y, z = Var("x"), Var("y"), Var("z")
term = Add(Mul(x, y), Mul(x, z))
language = ArithmeticLanguage()
config = MCTSConfig(iterations=100, rollout_depth=12, seed=0)

best = rewrite(language, term, config)  # x * (y + z)
result = rewrite_detailed(language, term, config)
print(result.initial_cost, result.final_cost)  # 7.0, 5.0
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
src/
  api/                         public API and benchmark CLI
  language/                    Term, RewriteRule, Language
  core/                        Engine, EngineConfig, RewriteResult, TraceStep
  engines/
    mcts/                      configuration and hand-written search
    egglog/                    optional engine and local adapter protocol
benchmarks/
  common.py                    benchmark metadata and tree batch-rewrite helper
  registry.py                  explicit suite discovery and lookup
  arithmetic/
  boolean/
  lambda/
  bitvector/
  linear_algebra/
  regex/
tests/                         engine, semantics, registry, CLI, and egglog tests
```

Each suite contains its own immutable terms, rules, costs, and benchmark definitions.
The shared tree traversal helper belongs to `benchmarks/`; the core and engines
never inspect an AST, node operator, child list, binder, or matrix dimension.

## Search

One MCTS action is whatever the language returns from `applicable(term)`.
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

Default configuration:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `iterations` | 1000 | Maximum search iterations; zero returns the input |
| `rollout_depth` | 12 | Maximum random transitions after expansion |
| `exploration_constant` | sqrt(2) | UCT exploration weight |
| `seed` | 0 | Local RNG seed; use `None` for unseeded runs |
| `max_tree_depth` | 64 | Maximum selection/expansion depth per iteration |
| `max_states` | 10000 | Bound on cached unique states |
| `time_limit` | None | Optional elapsed-time limit in seconds |

Statistics include iterations, states explored, first best iteration,
transpositions, cycles, transition evaluations, rollout steps, and stop reason.
Time limits are checked between transitions; a language action already executing
runs to completion. With iteration/state budgets, deterministic languages, and a
fixed seed, terms, traces, and search statistics are reproducible. Wall time is
not deterministic, and elapsed-time budgets can change the search path.

This is a bounded research search, with no claim of globally optimal results.

## Languages and benchmarks

There are 27 hand-constructed benchmarks. Difficulty labels describe expected
optimization/search difficulty and never change engine behavior.

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

## Validation

`pytest` covers engine dispatch, non-tree and unhashable terms, seeded
reproducibility, cycles, transpositions, temporary cost increases, retained
intermediate states, budgets, deterministic batch matching, all suite lookups,
all benchmark smoke runs, CLI handling, and supported egglog comparisons.
Semantic checks exercise arithmetic values, Boolean truth tables, bitvector
wraparound, lambda capture avoidance, matrix values/shapes, and regex membership.

Egglog-dependent tests skip cleanly when the optional package is absent.
CI runs the full suite, lint checks, and aggregate CLI on Python 3.11, 3.12, and 3.14.
