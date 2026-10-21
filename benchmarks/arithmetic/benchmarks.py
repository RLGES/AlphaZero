from benchmarks.common import Benchmark, Suite

from .language import ArithmeticLanguage
from .term import Add, Const, Mul, Var

x, y, z = (Var(name) for name in ("x", "y", "z"))

SUITE = Suite(
    "arithmetic",
    ArithmeticLanguage(),
    (
        Benchmark("factor_once", Add(Mul(x, y), Mul(x, z)), "easy", "Share a common factor."),
        Benchmark(
            "constant_chain",
            Add(Add(x, Const(2)), Const(3)),
            "medium",
            "Reassociate before folding constants.",
        ),
        Benchmark(
            "expand_to_fold",
            Mul(Add(x, Const(0)), Add(Const(2), Const(3))),
            "medium",
            "Branching between simplification and distribution.",
        ),
        Benchmark(
            "expand_cancel",
            Add(Mul(Const(2), Add(x, Const(3))), Mul(Const(-2), x)),
            "hard",
            "Expand through a cost increase to expose cancellation.",
        ),
        Benchmark(
            "hidden_factor",
            Add(Add(Mul(x, y), Mul(z, x)), Mul(x, Const(0))),
            "hard",
            "Remove noise, expose a nonadjacent common factor, and refactor.",
        ),
    ),
)
