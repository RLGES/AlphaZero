"""Small benchmark CLI, also usable as python -m src.api.cli."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Sequence

from benchmarks.registry import SUITE_NAMES, all_benchmarks, load_suite, lookup
from src.core import RewriteResult
from src.engines.alphazero import AlphaZeroConfig
from src.engines.egglog import EgglogConfig, EgglogUnavailableError, UnsupportedLanguageError

from .rewrite import rewrite_detailed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rewrite",
        description="Optimize a benchmark with hand-written MCTS.",
        epilog="Examples: rewrite arithmetic/factor_once; rewrite suite boolean; rewrite all",
    )
    parser.add_argument("target", help="<suite>/<benchmark>, suite, all, or list")
    parser.add_argument("suite_name", nargs="?", choices=SUITE_NAMES)
    parser.add_argument("--engine", choices=("mcts", "alphazero"), default="mcts")
    parser.add_argument("--checkpoint", help="AlphaZero network checkpoint (default: untrained)")
    parser.add_argument("--compare-egglog", action="store_true")
    parser.add_argument("--iterations", type=int, help="Override MCTS iterations")
    parser.add_argument("--seed", type=int, help="Override the MCTS seed")
    parser.add_argument("--verbose", action="store_true", help="Print terms and available traces")
    return parser


def _details(suite, result: RewriteResult) -> None:
    print(f"  Initial term: {suite.language.format(result.initial_term)}")
    print(f"  Final term:   {suite.language.format(result.final_term)}")
    if result.trace:
        print("  Rewrite trace:")
        for i, step in enumerate(result.trace, 1):
            print(f"    {i}. {step.action}: {step.before_cost:g} -> {step.after_cost:g}")
            print(f"       {suite.language.format(step.after)}")
    elif result.engine == "egglog":
        print("  Trace: unavailable for equality saturation.")
    print(f"  Statistics: {result.statistics}")


def _single(name: str, difficulty: str, result: RewriteResult) -> None:
    print(f"Benchmark: {name}\nDifficulty: {difficulty}\nEngine: {result.engine}\n")
    print(f"Initial cost: {result.initial_cost:g}")
    print(f"Final cost:   {result.final_cost:g}")
    print(f"Improvement:  {result.improvement_percent:.1f}%")
    print(f"Time:         {result.wall_time:.4f}s")
    print(f"Iterations:   {result.statistics['iterations']}")
    print(f"States seen:  {result.statistics['states_explored']}")
    print(f"Best found:   iteration {result.statistics['best_iteration']}")


def _row(result: RewriteResult) -> str:
    return (
        f"{result.engine:<8} {result.initial_cost:>9g} {result.final_cost:>9g} "
        f"{result.improvement_percent:>10.1f}% {result.wall_time:>9.4f}s"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.iterations is not None and args.iterations < 0:
        parser.error("--iterations must be nonnegative")
    if args.target != "suite" and args.suite_name is not None:
        parser.error("A suite name is only accepted after 'suite'")
    try:
        if args.target == "list":
            for suite, benchmark in all_benchmarks():
                print(
                    f"{suite.name}/{benchmark.name:<24} {benchmark.difficulty:<6} "
                    f"{benchmark.description}"
                )
            return 0
        if args.target == "suite":
            if args.suite_name is None:
                parser.error("'suite' requires a suite name")
            suite = load_suite(args.suite_name)
            selected = [(suite, benchmark) for benchmark in suite.benchmarks]
        elif args.target == "all":
            selected = all_benchmarks()
        else:
            selected = [lookup(args.target)]
    except ValueError as exc:
        parser.error(str(exc))

    aggregate = args.target in ("suite", "all")
    if aggregate:
        print(
            f"{'Benchmark':<42} {'Level':<6} {'Engine':<8} {'Initial':>9} "
            f"{'Final':>9} {'Improvement':>11} {'Time':>10}"
        )
    for suite, benchmark in selected:
        config = benchmark.config
        if args.engine == "alphazero":
            config = AlphaZeroConfig(
                iterations=config.iterations, seed=config.seed, checkpoint=args.checkpoint
            )
        if args.iterations is not None:
            config = replace(config, iterations=args.iterations)
        if args.seed is not None:
            config = replace(config, seed=args.seed)
        result = rewrite_detailed(suite.language, benchmark.term, config)
        name = f"{suite.name}/{benchmark.name}"
        results = [result]
        comparison_note = None
        if args.compare_egglog:
            try:
                results.append(rewrite_detailed(suite.language, benchmark.term, EgglogConfig()))
            except (UnsupportedLanguageError, EgglogUnavailableError) as exc:
                comparison_note = f"egglog comparison skipped: {exc}"

        if aggregate:
            for item in results:
                print(f"{name:<42} {benchmark.difficulty:<6} {_row(item)}")
        elif args.compare_egglog:
            print(f"Benchmark: {name}\nDifficulty: {benchmark.difficulty}\n")
            print(f"{'Engine':<8} {'Initial':>9} {'Final':>9} {'Improvement':>11} {'Time':>10}")
            for item in results:
                print(_row(item))
            print(
                f"MCTS iterations: {result.statistics['iterations']}; "
                f"states seen: {result.statistics['states_explored']}; "
                f"best found: iteration {result.statistics['best_iteration']}"
            )
        else:
            _single(name, benchmark.difficulty, result)
        if comparison_note:
            print(f"  {name}: {comparison_note}")
        if args.verbose:
            for item in results:
                print(f"  {item.engine}:")
                _details(suite, item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
