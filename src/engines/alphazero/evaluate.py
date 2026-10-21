"""Compare MCTS, AlphaZero and egglog on held-out terms.

    python -m src.engines.alphazero.evaluate --suite arithmetic \
        --checkpoint checkpoints/arith/latest.pt --budgets 10,25,50,100 --seeds 3 --generated 30

Iterations are not an equal budget across engines (an MCTS iteration includes
a random rollout), so the table also reports ``transitions`` (calls to
``language.apply``) and wall time. "best-known" is the lowest cost any engine
reached on a term in this evaluation.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Sequence
from random import Random

from benchmarks.generators import GENERATOR_SUITES, generate
from benchmarks.registry import load_suite
from src.api import rewrite_detailed
from src.engines.egglog import EgglogConfig, EgglogUnavailableError, UnsupportedLanguageError
from src.engines.mcts import MCTSConfig

from .config import AlphaZeroConfig

# Training draws terms from seeds near zero; this one never collides with them.
TEST_SEED = 987_654_321


def _rows(args: argparse.Namespace) -> list[dict]:
    suite = load_suite(args.suite)
    language = suite.language
    terms = [(f"{suite.name}/{b.name}", b.term) for b in suite.benchmarks]
    if args.generated:
        generated = generate(args.suite, Random(TEST_SEED), args.generated, (3, 5))
        terms += [(f"generated/{i:03d}", term) for i, term in enumerate(generated)]
    network = None
    if args.checkpoint:
        from .checkpoint import load_network

        network = load_network(args.checkpoint, language, args.device)

    rows = []
    for name, term in terms:
        for budget in args.budgets:
            for seed in range(args.seeds):
                configs = (
                    MCTSConfig(iterations=budget, seed=seed),
                    AlphaZeroConfig(iterations=budget, seed=seed, network=network),
                )
                for config in configs:
                    result = rewrite_detailed(language, term, config)
                    rows.append(_row(name, budget, seed, result))
        if args.egglog:
            try:
                result = rewrite_detailed(language, term, EgglogConfig())
                rows.append(_row(name, 0, 0, result))
            except (UnsupportedLanguageError, EgglogUnavailableError):
                pass
    best = defaultdict(lambda: float("inf"))
    for row in rows:
        best[row["term"]] = min(best[row["term"]], row["final_cost"])
    for row in rows:
        row["best_known"] = row["final_cost"] <= best[row["term"]]
    return rows


def _row(name: str, budget: int, seed: int, result) -> dict:
    stats = result.statistics
    return {
        "term": name,
        "engine": result.engine,
        "budget": budget,
        "seed": seed,
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "improvement": result.improvement_percent,
        "transitions": stats.get("transitions", ""),
        "states": stats.get("states_explored", ""),
        "seconds": result.wall_time,
        "best_known": False,
    }


def _summary(rows: list[dict]) -> None:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["engine"], row["budget"])].append(row)
    print(
        f"{'Engine':<10} {'Budget':>6} {'Runs':>5} {'Improvement':>12} {'best-known':>11} "
        f"{'transitions':>12} {'Seconds':>9}"
    )
    for (engine, budget), group in sorted(
        groups.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        transitions = [r["transitions"] for r in group if r["transitions"] != ""]
        mean_transitions = sum(transitions) / len(transitions) if transitions else float("nan")
        print(
            f"{engine:<10} {budget or '-':>6} {len(group):>5} "
            f"{sum(r['improvement'] for r in group) / len(group):>11.1f}% "
            f"{100 * sum(r['best_known'] for r in group) / len(group):>10.0f}% "
            f"{mean_transitions:>12.1f} {sum(r['seconds'] for r in group) / len(group):>9.4f}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero against MCTS and egglog.")
    parser.add_argument("--suite", choices=GENERATOR_SUITES, required=True)
    parser.add_argument("--checkpoint", help="Trained network (default: untrained AlphaZero)")
    parser.add_argument(
        "--budgets", type=lambda s: [int(x) for x in s.split(",")], default=[10, 25, 50, 100]
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--generated", type=int, default=0, help="Extra held-out random terms")
    parser.add_argument("--no-egglog", dest="egglog", action="store_false")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv", help="Write one row per run to this file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = _rows(args)
    _summary(rows)
    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
