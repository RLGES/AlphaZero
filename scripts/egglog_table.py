"""Trained AlphaZero and MCTS vs the egglog equality-saturation baseline.

    .venv/bin/python scripts/egglog_table.py --set heldout \
        builtin='none:checkpoints/asm_v4_builtin_s*' llm='learned_rules_qwen_all:checkpoints/asm_v4_llm_s*'

Each arm is ``label='<rules label>:<run glob>'``; the rules label names the egglog
cache ``checkpoints/final_egglog_<rules>_<set>.csv`` written by final_eval. For every
arm and budget: AlphaZero (mean ± std over seeds), MCTS, egglog, and the paired
per-term difference AlphaZero − egglog with a 95% bootstrap interval.
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

CHECKPOINTS = Path("checkpoints")


def _rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def by_term(rows: list[dict], engine: str, budget: int) -> dict[str, float]:
    return {
        row["term"]: float(row["improvement"])
        for row in rows
        if row["engine"] == engine and int(row["budget"]) == budget
    }


def interval(diffs: list[float], rng: Random) -> str:
    boots = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(2000))
    return f"{statistics.mean(diffs):+.1f} [{boots[50]:+.1f}, {boots[1949]:+.1f}]"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", default="heldout")
    parser.add_argument("arms", nargs="+", help="label='<rules label>:<run glob>'")
    args = parser.parse_args(argv)
    rng = Random(0)
    print("| Arm | Seeds | Budget | Trained AZ | MCTS | egglog | AZ − egglog (95% CI) |")
    print("|---|---|---|---|---|---|---|")
    for spec in args.arms:
        label, rest = spec.split("=", 1)
        rules, pattern = rest.split(":", 1)
        cache = CHECKPOINTS / f"final_egglog_{rules}_{args.set}.csv"
        csv_name = f"final_{args.set}.csv"
        runs = sorted(Path(p) for p in glob.glob(pattern) if (Path(p) / csv_name).exists())
        if not cache.exists() or not runs:
            print(f"{label}: missing {cache} or finished runs for {pattern}", file=sys.stderr)
            continue
        egglog = by_term(_rows(cache), "egglog", 0)
        tables = [_rows(run / csv_name) for run in runs]
        budgets = sorted({int(r["budget"]) for r in tables[0] if r["engine"] == "AlphaZero"})
        for budget in budgets:
            az = [by_term(rows, "AlphaZero", budget) for rows in tables]
            mcts = by_term(tables[0], "MCTS", budget)
            terms = sorted(set(egglog).intersection(*az))
            means = [statistics.mean(seed[t] for t in terms) for seed in az]
            per_term = defaultdict(list)
            for seed in az:
                for t in terms:
                    per_term[t].append(seed[t])
            diffs = [statistics.mean(per_term[t]) - egglog[t] for t in terms]
            spread = f" ± {statistics.stdev(means):.1f}" if len(means) > 1 else ""
            print(
                f"| {label} | {len(runs)} | {budget} | {statistics.mean(means):.1f}{spread} "
                f"| {statistics.mean(mcts[t] for t in terms):.1f} "
                f"| {statistics.mean(egglog[t] for t in terms):.1f} | {interval(diffs, rng)} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
