from benchmarks.common import Benchmark, Suite

from .language import LinearAlgebraLanguage
from .term import Add, Identity, MatMul, Matrix, Transpose, Zero

a, b, c = Matrix("A", 64, 8), Matrix("B", 8, 256), Matrix("C", 256, 2)
p, q, r = (Matrix(name, 8, 8) for name in ("P", "Q", "R"))

SUITE = Suite(
    "linear_algebra",
    LinearAlgebraLanguage(),
    (
        Benchmark("identity", MatMul(Identity(8), p), "easy", "Remove a matrix identity."),
        Benchmark(
            "chain_order",
            MatMul(MatMul(a, b), c),
            "medium",
            "Choose parentheses that reduce scalar multiplication work.",
        ),
        Benchmark(
            "common_left",
            Add(MatMul(p, q), MatMul(p, r)),
            "medium",
            "Share a left factor without commuting matrix products.",
        ),
        Benchmark(
            "transpose_chain",
            Transpose(MatMul(MatMul(a, b), c)),
            "hard",
            "Reassociate and push a transpose through a product.",
        ),
        Benchmark(
            "zero_branch",
            Add(MatMul(p, Zero(8, 8)), Transpose(Transpose(q))),
            "easy",
            "Remove a zero product and two transposes.",
        ),
    ),
)
