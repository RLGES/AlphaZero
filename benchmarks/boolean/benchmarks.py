from benchmarks.common import Benchmark, Suite

from .language import BooleanLanguage
from .term import And, Bool, Not, Or, Var

p, q, r = (Var(name) for name in ("p", "q", "r"))

SUITE = Suite(
    "boolean",
    BooleanLanguage(),
    (
        Benchmark(
            "demorgan", Not(And(Not(p), Not(q))), "easy", "De Morgan exposes two double negations."
        ),
        Benchmark("absorption", Or(p, And(p, q)), "easy", "Absorb a redundant branch."),
        Benchmark(
            "complement_factor",
            Or(And(p, q), And(p, Not(q))),
            "medium",
            "Factor before recognizing a complementary pair.",
        ),
        Benchmark(
            "hidden_absorption",
            And(Or(p, q), Or(p, And(Not(q), Bool(True)))),
            "hard",
            "Distribution can initially increase cost before complement elimination.",
        ),
    ),
)
