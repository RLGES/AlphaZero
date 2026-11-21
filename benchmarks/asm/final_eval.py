"""Final evaluation of trained asm networks on every held-out term and a stress set.

    RLGES_ASM_LEARNED=none .venv/bin/python -m benchmarks.asm.final_eval \
        --runs 'checkpoints/asm_v3_builtin_s*' --workers 2

The rule set comes from ``RLGES_ASM_LEARNED`` (as in training); every run must have
been trained with it. Test sets:

* ``heldout``: all of ``test.jsonl`` + ``cf_test.jsonl`` (terms training never draws)
* ``stress``: ``stress_test.jsonl`` (longer programs and larger terms than training),
  minus any term that also appears in a corpus file

MCTS does not depend on the network, so it runs once per rule set and test set
(cached in ``checkpoints/final_mcts_<rules>_<set>.csv``). Each run gets
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


def _evaluate(job: tuple[str, Expr, str | None, tuple[int, ...]]) -> list[dict]:
    """All budgets for one term: MCTS when checkpoint is None, else AlphaZero."""
    from src.api import rewrite_detailed
    from src.engines.alphazero.config import AlphaZeroConfig
    from src.engines.mcts import MCTSConfig

    from .language import AsmLanguage

    name, term, checkpoint, budgets = job
    language = AsmLanguage()
    rows = []
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
        for run in runs:
            out = run / f"final_{set_name}{tag}.csv"
            if out.exists():
                log(f"{set_name}: {run.name} already done")
                continue
            log(f"{set_name}: {run.name} AlphaZero on {len(terms)} terms")
            jobs = [(n, t, str(run / "best.pt"), budgets) for n, t in terms]
            _write(out, _run_jobs(jobs, workers) + mcts)


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
