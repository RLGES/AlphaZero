from benchmarks.common import Benchmark, Suite

from .language import LambdaLanguage
from .term import Abs, App, Var

x, y, f = Var("x"), Var("y"), Var("f")
identity = Abs("x", x)
redundant_identity = Abs("u", App(identity, App(identity, Var("u"))))

SUITE = Suite(
    "lambda",
    LambdaLanguage(),
    (
        Benchmark(
            "nested_beta",
            App(identity, App(identity, y)),
            "easy",
            "Two nested identity applications.",
        ),
        Benchmark("eta", Abs("x", App(f, x)), "easy", "Remove a redundant abstraction."),
        Benchmark(
            "capture_avoidance",
            App(Abs("x", Abs("y", x)), y),
            "medium",
            "Beta reduction must rename a binder rather than capture free y.",
        ),
        Benchmark(
            "duplicate_then_reduce",
            App(Abs("f", App(f, App(f, y))), redundant_identity),
            "hard",
            "Duplicating a function initially increases cost before beta reduction.",
        ),
    ),
)
