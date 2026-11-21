"""Trained AlphaZero and MCTS vs the egglog equality-saturation baseline and hybrids.

    .venv/bin/python scripts/egglog_table.py --set heldout \
        builtin='none:checkpoints/asm_v4_builtin_s*' llm='learned_rules_qwen_all:checkpoints/asm_v4_llm_s*'

Each arm is ``label='<rules label>:<run glob>'``; the rules label names the caches
``checkpoints/final_egglog_<rules>_<set>.csv`` and ``final_egglogmcts_<rules>_<set>.csv``
written by final_eval. For every arm and budget: AlphaZero (mean ± std over seeds),
MCTS, egglog, egglog + MCTS, egglog + AlphaZero (``<run>/final_<set>_hybrid.csv``),
and paired per-term differences with 95% bootstrap intervals: AlphaZero alone vs
egglog, egglog + AlphaZero vs egglog (does combining help?), and egglog + AlphaZero vs
egglog + MCTS (is it the trained policy, or just more search?). Hybrid columns read
"—" until their files exist.
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
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


def seed_mean(seeds: list[dict[str, float]], terms: list[str]) -> tuple[str, dict[str, float]]:
    """("mean ± std over seeds", per-term mean over seeds)."""
    means = [statistics.mean(seed[t] for t in terms) for seed in seeds]
    spread = f" ± {statistics.stdev(means):.1f}" if len(means) > 1 else ""
    per_term = {t: statistics.mean(seed[t] for seed in seeds) for t in terms}
    return f"{statistics.mean(means):.1f}{spread}", per_term


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", default="heldout")
    parser.add_argument("arms", nargs="+", help="label='<rules label>:<run glob>'")
    args = parser.parse_args(argv)
    rng = Random(0)
    print(
        "| Arm | Seeds | Budget | Trained AZ | MCTS | egglog | egglog+MCTS | egglog+AZ "
        "| AZ − egglog (95% CI) | egglog+AZ − egglog (95% CI) | egglog+AZ − egglog+MCTS (95% CI) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for spec in args.arms:
        label, rest = spec.split("=", 1)
        rules, pattern = rest.split(":", 1)
        cache = CHECKPOINTS / f"final_egglog_{rules}_{args.set}.csv"
        hybrid_cache = CHECKPOINTS / f"final_egglogmcts_{rules}_{args.set}.csv"
        csv_name = f"final_{args.set}.csv"
        runs = sorted(Path(p) for p in glob.glob(pattern) if (Path(p) / csv_name).exists())
        if not cache.exists() or not runs:
            print(f"{label}: missing {cache} or finished runs for {pattern}", file=sys.stderr)
            continue
        egglog = by_term(_rows(cache), "egglog", 0)
        tables = [_rows(run / csv_name) for run in runs]
        hybrid_paths = [run / f"final_{args.set}_hybrid.csv" for run in runs]
        hybrids = [_rows(p) for p in hybrid_paths] if all(p.exists() for p in hybrid_paths) else []
        mcts_hybrid = _rows(hybrid_cache) if hybrid_cache.exists() else []
        budgets = sorted({int(r["budget"]) for r in tables[0] if r["engine"] == "AlphaZero"})
        for budget in budgets:
            az = [by_term(rows, "AlphaZero", budget) for rows in tables]
            mcts = by_term(tables[0], "MCTS", budget)
            terms = sorted(set(egglog).intersection(*az))
            az_text, az_term = seed_mean(az, terms)
            cells = [
                f"{statistics.mean(mcts[t] for t in terms):.1f}",
                f"{statistics.mean(egglog[t] for t in terms):.1f}",
            ]
            em = by_term(mcts_hybrid, "egglog+MCTS", budget)
            cells.append(f"{statistics.mean(em[t] for t in terms):.1f}" if em else "—")
            hybrid_cell = gain_cell = policy_cell = "—"
            if hybrids:
                ea = [by_term(rows, "egglog+AlphaZero", budget) for rows in hybrids]
                hybrid_cell, ea_term = seed_mean(ea, terms)
                gain_cell = interval([ea_term[t] - egglog[t] for t in terms], rng)
                if em:  # does the trained policy beat plain search from the same start?
                    policy_cell = interval([ea_term[t] - em[t] for t in terms], rng)
            cells.append(hybrid_cell)
            az_gap = interval([az_term[t] - egglog[t] for t in terms], rng)
            print(
                f"| {label} | {len(runs)} | {budget} | {az_text} | "
                + " | ".join(cells)
                + f" | {az_gap} | {gain_cell} | {policy_cell} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
