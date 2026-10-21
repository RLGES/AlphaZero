"""Explicit discovery keeps suite imports predictable, including Python's lambda keyword."""

from functools import lru_cache
from importlib import import_module

from .common import Benchmark, Suite

SUITE_NAMES = ("arithmetic", "boolean", "lambda", "bitvector", "linear_algebra", "regex", "asm")


@lru_cache(maxsize=None)
def load_suite(name: str) -> Suite:
    if name not in SUITE_NAMES:
        raise ValueError(f"Unknown suite '{name}'. Choose from: {', '.join(SUITE_NAMES)}")
    suite = import_module(f"benchmarks.{name}.benchmarks").SUITE
    names = [benchmark.name for benchmark in suite.benchmarks]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate benchmark names in {name}")
    return suite


def lookup(identifier: str) -> tuple[Suite, Benchmark]:
    parts = identifier.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Benchmark names must have the form <suite>/<benchmark>")
    suite = load_suite(parts[0])
    for benchmark in suite.benchmarks:
        if benchmark.name == parts[1]:
            return suite, benchmark
    choices = ", ".join(benchmark.name for benchmark in suite.benchmarks)
    raise ValueError(f"Unknown benchmark '{identifier}'. Available in {suite.name}: {choices}")


def all_benchmarks() -> list[tuple[Suite, Benchmark]]:
    return [
        (suite, benchmark)
        for name in SUITE_NAMES
        for suite in [load_suite(name)]
        for benchmark in suite.benchmarks
    ]
