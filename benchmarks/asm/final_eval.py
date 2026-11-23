"""Final evaluation of trained asm networks on every held-out term and a stress set.

    RLGES_ASM_LEARNED=none .venv/bin/python -m benchmarks.asm.final_eval \
        --runs 'checkpoints/asm_v3_builtin_s*' --workers 2

The rule set comes from ``RLGES_ASM_LEARNED`` (as in training); every run must have
been trained with it. Test sets:

* ``heldout``: all of ``test.jsonl`` + ``cf_test.jsonl`` (terms training never draws)
* ``stress``: ``stress_test.jsonl`` (longer programs and larger terms than training),
  minus any term that also appears in a corpus file

MCTS does not depend on the network, so it runs once per rule set and test set
(cached in ``checkpoints/final_mcts_<rules>_<set>.csv``); so does the egglog
equality-saturation baseline (``checkpoints/final_egglog_<rules>_<set>.csv``, one
row per term with budget 0; ``scripts/egglog_table.py`` compares it). Hybrids start
the search from egglog's extracted term: egglog + MCTS once per rule set
(``final_egglogmcts_<rules>_<set>.csv``) and egglog + AlphaZero per run
(``<run>/final_<set>_hybrid.csv``); their costs count from the original term. Each run gets
``<run>/final_<set>.csv`` with its AlphaZero rows plus the cached MCTS rows, in the
``eval_after.csv`` format, so ``scripts/ablation_table.py --csv final_<set>.csv``
builds the table. Finished files are kept, so a rerun only does what is missing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from .generator import CORPUS_DIR
from .learned import learned_path
from .term import Expr, from_sexpr

FIELDS = (
    "term",
    "engine",
    "budget",
    "seed",
    "initial_cost",
    "final_cost",
    "improvement",
    "transitions",
    "states",
    "seconds",
)
CHECKPOINTS = Path("checkpoints")


def _read(name: str) -> list[Expr]:
    path = CORPUS_DIR / name
    if not path.exists():
        return []
    return [from_sexpr(json.loads(line)["sexpr"]) for line in path.open()]


def test_sets() -> dict[str, list[tuple[str, Expr]]]:
    """name -> [(term id, term)], deduplicated; stress excludes every corpus term."""
    heldout = list(dict.fromkeys(_read("test.jsonl") + _read("cf_test.jsonl")))
    corpus = set(heldout) | set(_read("train.jsonl")) | set(_read("cf_train.jsonl"))
    stress = [t for t in dict.fromkeys(_read("stress_test.jsonl")) if t not in corpus]
    return {
        "heldout": [(f"heldout/{i:04d}", t) for i, t in enumerate(heldout)],
        "stress": [(f"stress/{i:04d}", t) for i, t in enumerate(stress)],
    }


def rules_label() -> str:
    path = learned_path()
    return "none" if path is None else Path(path).stem


def _summary_rules(run: Path) -> str | None:
    """The rule set a run was trained with, from its overnight summary."""
    summary = CHECKPOINTS / f"summary_{run.name}.txt"
    if not summary.exists():
        return None
    first = summary.read_text().splitlines()[0]
    return first.split(", rules ", 1)[1].split(" (", 1)[0] if ", rules " in first else None


@lru_cache(maxsize=None)
def _network(checkpoint: str):
    import torch

    from src.engines.alphazero.checkpoint import load_network

    from .language import AsmLanguage

    torch.set_num_threads(1)
    return load_network(checkpoint, AsmLanguage())


def _egglog_guarded(language, term, rules: set[str] | None = None):
    """egglog in a forked child with a memory cap (EGGLOG_MEM_GB, default 2) and a
    timeout (EGGLOG_TIMEOUT s, default 300). With many learned rules the e-graph can
    outgrow the machine (one held-out term reached 11.7 GB and the OOM killer took
    the whole pool), so on failure it retries with fewer iterations (6, 4, 2, 1) and,
    if all fail, returns the input term unchanged. Fallbacks go to
    checkpoints/egglog_fallbacks.log. Returns (final_term, final_cost, seconds, engine)."""
    import pickle
    import resource
    import select
    import time

    from src.api import rewrite_detailed
    from src.engines.egglog import EgglogConfig

    limit = int(float(os.environ.get("EGGLOG_MEM_GB", "2")) * 2**30)
    timeout = float(os.environ.get("EGGLOG_TIMEOUT", "300"))
    start = time.perf_counter()
    for iterations in (6, 4, 2, 1):
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                if rules is not None:
                    os.environ["RLGES_EGGLOG_RULES"] = ",".join(sorted(rules))
                r = rewrite_detailed(language, term, EgglogConfig(iterations=iterations))
                matches = r.statistics.get("rule_matches", "")
                data = pickle.dumps((r.final_term, r.final_cost, r.engine, matches))
                with os.fdopen(write, "wb") as out:
                    out.write(data)
            finally:
                os._exit(0)
        os.close(write)
        chunks, deadline = [], time.monotonic() + timeout
        with os.fdopen(read, "rb") as inp:
            while True:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([inp], [], [], left)[0]:
                    os.kill(pid, 9)
                    break
                chunk = inp.read1(1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
        os.waitpid(pid, 0)
        if chunks:
            try:
                final_term, final_cost, engine, matches = pickle.loads(b"".join(chunks))
            except Exception:
                final_term = None
            if final_term is not None:
                if iterations != 6:
                    _note_fallback(term, f"egglog ok with iterations={iterations}")
                _LAST_MATCHES[0] = matches
                return final_term, final_cost, time.perf_counter() - start, engine
    _note_fallback(term, "egglog failed at every iteration count; input kept")
    _LAST_MATCHES[0] = ""
    return term, language.cost(term), time.perf_counter() - start, "egglog"


_LAST_MATCHES = [""]  # rule matches of the last guarded egglog call (e-graph growth proxy)


def _note_fallback(term, message: str) -> None:
    with (CHECKPOINTS / "egglog_fallbacks.log").open("a") as log:
        log.write(f"{os.environ.get('RLGES_ASM_LEARNED', '')}\t{message}\t{term}\n")


def _evaluate(job: tuple[str, Expr, str | None, tuple[int, ...]]) -> list[dict]:
    """All budgets for one term. ``checkpoint`` picks the engine: None is MCTS,
    "egglog" is egglog (one row, budget 0), "egglog+mcts" and "egglog+az:<path>"
    search from egglog's result, and any other value is an AlphaZero checkpoint."""
    from src.api import rewrite_detailed
    from src.engines.alphazero.config import AlphaZeroConfig
    from src.engines.mcts import MCTSConfig

    from .language import AsmLanguage

    name, term, checkpoint, budgets = job
    language = AsmLanguage()
    rows = []
    if checkpoint in ("egglog+mcts",) or (checkpoint or "").startswith("egglog+az:"):
        return _hybrid(name, term, checkpoint, budgets)
    if (checkpoint or "").startswith(("az+egglog:", "azrules+egglog:")):
        return _az_first(name, term, checkpoint, budgets)
    if checkpoint == "egglog":
        budgets = (0,)
    if checkpoint == "egglog":
        initial = language.cost(term)
        _, final, seconds, engine = _egglog_guarded(language, term)
        return [
            {
                "term": name,
                "engine": engine,
                "budget": 0,
                "seed": 0,
                "initial_cost": initial,
                "final_cost": final,
                "improvement": 100 * (initial - final) / initial if initial else 0.0,
                "transitions": "",
                "states": _LAST_MATCHES[0],
                "seconds": seconds,
            }
        ]
    for budget in budgets:
        if checkpoint is None:
            config = MCTSConfig(iterations=budget, seed=0)
        else:
            config = AlphaZeroConfig(iterations=budget, seed=0, network=_network(checkpoint))
        result = rewrite_detailed(language, term, config)
        stats = result.statistics
        rows.append(
            {
                "term": name,
                "engine": result.engine,
                "budget": budget,
                "seed": 0,
                "initial_cost": result.initial_cost,
                "final_cost": result.final_cost,
                "improvement": result.improvement_percent,
                "transitions": stats.get("transitions", ""),
                "states": stats.get("states_explored", ""),
                "seconds": result.wall_time,
            }
        )
    return rows


def _hybrid(name: str, term: Expr, checkpoint: str, budgets: tuple[int, ...]) -> list[dict]:
    """egglog first, then MCTS or AlphaZero from its extracted term."""
    from src.api import rewrite_detailed
    from src.engines.alphazero.config import AlphaZeroConfig
    from src.engines.mcts import MCTSConfig

    from .language import AsmLanguage

    language = AsmLanguage()
    first_term, first_cost, first_seconds, _ = _egglog_guarded(language, term)
    initial = language.cost(term)
    engine = "egglog+MCTS" if checkpoint == "egglog+mcts" else "egglog+AlphaZero"
    rows = []
    for budget in budgets:
        if checkpoint == "egglog+mcts":
            config = MCTSConfig(iterations=budget, seed=0)
        else:
            path = checkpoint.split(":", 1)[1]
            config = AlphaZeroConfig(iterations=budget, seed=0, network=_network(path))
        second = rewrite_detailed(language, first_term, config)
        final = min(second.final_cost, first_cost)
        rows.append(
            {
                "term": name,
                "engine": engine,
                "budget": budget,
                "seed": 0,
                "initial_cost": initial,
                "final_cost": final,
                "improvement": 100 * (initial - final) / initial if initial else 0.0,
                "transitions": second.statistics.get("transitions", ""),
                "states": second.statistics.get("states_explored", ""),
                "seconds": first_seconds + second.wall_time,
            }
        )
    return rows


def _az_rule_choice(language, term, path: str, trace) -> set[str]:
    """Rules AlphaZero picks for ``term``: the top AZRULES_K (default 24) by the
    network's prior at the root, plus every rule on AZ's best rewrite path."""
    from src.engines.alphazero.inference import NetworkEvaluator

    from .rules import PATTERN_RULES

    k = int(os.environ.get("AZRULES_K", "24"))
    ((prior, _),) = NetworkEvaluator(_network(path)).evaluate(language, [term])
    vocabulary = language.action_vocabulary()
    # AZRULES_SCOPE=learned (default) keeps every built-in rule and lets AZ choose
    # only among the learned ones; "all" lets AZ choose among every rule.
    builtin = {name for name, _, _ in PATTERN_RULES}
    scope_all = os.environ.get("AZRULES_SCOPE", "learned") == "all"
    pool = [i for i in range(len(vocabulary)) if scope_all or vocabulary[i].name not in builtin]
    ranked = sorted(pool, key=lambda i: -float(prior[i]))
    chosen = {vocabulary[i].name for i in ranked[:k]}
    if not scope_all:
        chosen |= builtin
    return chosen | {step.action.name for step in trace}


def _az_first(name: str, term: Expr, checkpoint: str, budgets: tuple[int, ...]) -> list[dict]:
    """AlphaZero first. "az+egglog": egglog (all rules) from AZ's result.
    "azrules+egglog": egglog on the original term with only the rules AZ chose
    (_az_rule_choice). Final cost is the better of AZ and egglog; "states" holds
    egglog's rule matches (a proxy for e-graph growth)."""
    from src.api import rewrite_detailed
    from src.engines.alphazero.config import AlphaZeroConfig

    from .language import AsmLanguage

    language = AsmLanguage()
    kind, path = checkpoint.split(":", 1)
    initial = language.cost(term)
    rows = []
    for budget in budgets:
        config = AlphaZeroConfig(iterations=budget, seed=0, network=_network(path))
        az = rewrite_detailed(language, term, config)
        if kind == "az+egglog":
            _, egg_cost, egg_seconds, _ = _egglog_guarded(language, az.final_term)
            engine = "AlphaZero+egglog"
        else:
            rules = _az_rule_choice(language, term, path, az.trace)
            _, egg_cost, egg_seconds, _ = _egglog_guarded(language, term, rules)
            engine = f"AZrules+egglog(k={os.environ.get('AZRULES_K', '24')})"
        final = min(az.final_cost, egg_cost)
        rows.append(
            {
                "term": name,
                "engine": engine,
                "budget": budget,
                "seed": 0,
                "initial_cost": initial,
                "final_cost": final,
                "improvement": 100 * (initial - final) / initial if initial else 0.0,
                "transitions": az.statistics.get("transitions", ""),
                "states": _LAST_MATCHES[0],
                "seconds": az.wall_time + egg_seconds,
            }
        )
    return rows


def _run_jobs(jobs: list, workers: int) -> list[dict]:
    if workers <= 1:
        return [row for job in jobs for row in _evaluate(job)]
    with ProcessPoolExecutor(workers) as pool:
        return [row for rows in pool.map(_evaluate, jobs, chunksize=8) for row in rows]


def _write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _load(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def evaluate_runs(
    runs: list[Path],
    sets: list[str],
    budgets: tuple[int, ...],
    workers: int,
    log=print,
    limit: int = 0,
    tag: str = "",
) -> None:
    label = rules_label()
    wanted = os.environ.get("RLGES_ASM_LEARNED") or "<default learned_rules.json>"
    for run in runs:
        trained = _summary_rules(run)
        if trained is not None and trained != wanted:
            raise SystemExit(f"{run} was trained with rules {trained!r}, not {wanted!r}")
    available = test_sets()
    for set_name in sets:
        terms = available[set_name][:limit] if limit else available[set_name]
        if not terms:
            log(f"{set_name}: no terms (build stress_test.jsonl first); skipping")
            continue
        cache = CHECKPOINTS / f"final_mcts_{label}_{set_name}{tag}.csv"
        if not cache.exists():
            log(f"{set_name}: MCTS on {len(terms)} terms, rules {label}")
            _write(cache, _run_jobs([(n, t, None, budgets) for n, t in terms], workers))
        mcts = _load(cache)
        egglog_cache = CHECKPOINTS / f"final_egglog_{label}_{set_name}{tag}.csv"
        if not egglog_cache.exists():
            log(f"{set_name}: egglog on {len(terms)} terms, rules {label}")
            _write(egglog_cache, _run_jobs([(n, t, "egglog", budgets) for n, t in terms], workers))
        hybrid_cache = CHECKPOINTS / f"final_egglogmcts_{label}_{set_name}{tag}.csv"
        if not hybrid_cache.exists():
            log(f"{set_name}: egglog + MCTS on {len(terms)} terms, rules {label}")
            jobs = [(n, t, "egglog+mcts", budgets) for n, t in terms]
            _write(hybrid_cache, _run_jobs(jobs, workers))
        for run in runs:
            out = run / f"final_{set_name}{tag}.csv"
            if out.exists():
                log(f"{set_name}: {run.name} already done")
                continue
            log(f"{set_name}: {run.name} AlphaZero on {len(terms)} terms")
            jobs = [(n, t, str(run / "best.pt"), budgets) for n, t in terms]
            _write(out, _run_jobs(jobs, workers) + mcts)
        for run in runs:
            out = run / f"final_{set_name}_hybrid{tag}.csv"
            if out.exists():
                continue
            log(f"{set_name}: {run.name} egglog + AlphaZero on {len(terms)} terms")
            jobs = [(n, t, f"egglog+az:{run / 'best.pt'}", budgets) for n, t in terms]
            _write(out, _run_jobs(jobs, workers))


def evaluate_az_first(
    runs: list[Path], sets: list[str], budgets: tuple[int, ...], workers: int, log=print,
    limit: int = 0,
) -> None:
    """AZ -> egglog and AZ-chosen rules -> egglog for each run, written to
    <run>/final_<set>_azegg.csv and <run>/final_<set>_azrules.csv."""
    available = test_sets()
    for set_name in sets:
        terms = available[set_name][:limit] if limit else available[set_name]
        for run in runs:
            for kind, suffix in (("az+egglog", "azegg"), ("azrules+egglog", "azrules")):
                out = run / f"final_{set_name}_{suffix}{'_limit%d' % limit if limit else ''}.csv"
                if out.exists():
                    continue
                log(f"{set_name}: {run.name} {kind} on {len(terms)} terms")
                jobs = [(n, t, f"{kind}:{run / 'best.pt'}", budgets) for n, t in terms]
                _write(out, _run_jobs(jobs, workers))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", action="append", required=True, help="Run dir glob (repeatable)")
    parser.add_argument("--sets", default="heldout,stress")
    parser.add_argument("--budgets", default="10,25,50,100")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="First N terms per set (smoke)")
    args = parser.parse_args(argv)
    runs = sorted(
        {
            Path(p)
            for pattern in args.runs
            for p in glob.glob(pattern)
            if (Path(p) / "best.pt").exists() and (Path(p) / "report_after.txt").exists()
        }
    )
    if not runs:
        print("no finished runs match", args.runs, file=sys.stderr)
        return 1
    budgets = tuple(int(b) for b in args.budgets.split(","))
    evaluate_runs(
        runs,
        args.sets.split(","),
        budgets,
        args.workers,
        log=lambda line: print(line, flush=True),
        limit=args.limit,
        tag=f"_limit{args.limit}" if args.limit else "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
