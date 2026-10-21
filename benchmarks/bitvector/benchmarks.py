from benchmarks.common import Benchmark, Suite

from .language import BitvectorLanguage
from .term import Add, And, Const, Not, Or, Var, Xor

x, y = Var("x"), Var("y")

SUITE = Suite(
    "bitvector",
    BitvectorLanguage(),
    (
        Benchmark(
            "mask_identity",
            And(Xor(x, Const(0)), Const(255)),
            "easy",
            "Remove an empty XOR and an all-ones mask.",
        ),
        Benchmark(
            "wraparound",
            Add(Const(250), Const(10)),
            "easy",
            "Fold modulo 256 rather than unbounded integer arithmetic.",
        ),
        Benchmark(
            "xor_cancel",
            Xor(Xor(x, y), x),
            "medium",
            "Reassociation and commutation expose equal operands.",
        ),
        Benchmark(
            "demorgan_mask",
            Not(Or(Not(And(x, Const(255))), Not(y))),
            "hard",
            "Expose double complements across masks and Boolean identities.",
        ),
    ),
)
