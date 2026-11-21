"""Ablation table over training seeds from the eval_after.csv files of several runs.

    .venv/bin/python scripts/ablation_table.py builtin='checkpoints/asm_v3_builtin_s*' \
        llm='checkpoints/asm_v3_llm_s*'
    .venv/bin/python scripts/ablation_table.py --csv final_heldout.csv builtin=... llm=...

For every arm and budget: trained AlphaZero and MCTS improvement (mean ± std over
seeds), and the paired difference of each arm from the first one (same terms and
seeds in every arm) with a 95% bootstrap interval over terms.
"""

from __future__ import annotations

import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

CSV_NAME = "eval_after.csv"


def load(run: Path) -> dict[tuple[str, int], dict[tuple[str, int], float]]:
    """(engine, budget) -> {(term, eval seed): improvement}."""
    table: dict = defaultdict(dict)
    with (run / CSV_NAME).open() as handle:
        for row in csv.DictReader(handle):
            key = (row["engine"], int(row["budget"]))
            table[key][row["term"], int(row["seed"])] = float(row["improvement"])
    return table


def mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}"


def paired_interval(base: list[dict], other: list[dict], rng: Random) -> str:
    """Mean of (other - base) per term, averaged over the seeds both arms have."""
    seeds = min(len(base), len(other))
    terms = sorted(set.intersection(*(set(t) for t in base[:seeds] + other[:seeds])))
    if not terms:
        return "n/a"
    diffs = [
        statistics.mean(other[s][term] - base[s][term] for s in range(seeds)) for term in terms
    ]
    boots = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(2000))
    return f"{statistics.mean(diffs):+.1f} [{boots[50]:+.1f}, {boots[1949]:+.1f}]"


def main(argv: list[str]) -> int:
    global CSV_NAME
    if argv[:1] == ["--csv"]:
        CSV_NAME, argv = argv[1], argv[2:]
    arms = []
    for spec in argv:
        label, pattern = spec.split("=", 1)
        runs = sorted(Path(p) for p in glob.glob(pattern) if (Path(p) / CSV_NAME).exists())
        if runs:
            arms.append((label, [load(run) for run in runs]))
        else:
            print(f"{label}: no finished runs match {pattern}", file=sys.stderr)
    if not arms:
        return 1
    budgets = sorted({budget for _, runs in arms for _, budget in runs[0]})
    rng = Random(0)
    print(
        "| Arm | Seeds | Budget | Trained AZ | MCTS | AZ − MCTS | AZ Δ vs "
        + arms[0][0]
        + " (95% CI) |"
    )
    print("|---|---|---|---|---|---|---|")
    for label, runs in arms:
        for budget in budgets:
            az = [statistics.mean(run["AlphaZero", budget].values()) for run in runs]
            mcts = [statistics.mean(run["MCTS", budget].values()) for run in runs]
            gap = [a - m for a, m in zip(az, mcts)]
            delta = (
                "—"
                if label == arms[0][0]
                else paired_interval(
                    [run["AlphaZero", budget] for run in arms[0][1]],
                    [run["AlphaZero", budget] for run in runs],
                    rng,
                )
            )
            print(
                f"| {label} | {len(runs)} | {budget} | {mean_std(az)} | {mean_std(mcts)} "
                f"| {mean_std(gap)} | {delta} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
