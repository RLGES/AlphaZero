from benchmarks.common import Benchmark, Suite

from .language import RegexLanguage
from .term import Alt, Empty, Eps, Lit, Seq, Star

a, b, c = Lit("a"), Lit("b"), Lit("c")

SUITE = Suite(
    "regex",
    RegexLanguage(),
    (
        Benchmark(
            "epsilon",
            Seq(Eps(), Alt(a, Empty())),
            "easy",
            "Remove epsilon concatenation and empty alternation.",
        ),
        Benchmark(
            "common_prefix", Alt(Seq(a, b), Seq(a, c)), "medium", "Share a concatenation prefix."
        ),
        Benchmark(
            "nested_star",
            Star(Alt(Eps(), Star(a))),
            "easy",
            "Remove nullable noise and nested Kleene stars.",
        ),
        Benchmark(
            "expand_then_absorb",
            Seq(Alt(Eps(), a), Star(a)),
            "hard",
            "Distribution increases size before star absorption can simplify.",
        ),
        Benchmark(
            "factor_to_star",
            Alt(Seq(Star(a), b), Seq(Star(a), Empty())),
            "hard",
            "Simplify an annihilated branch while navigating expansion and factoring.",
        ),
    ),
)
