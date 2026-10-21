import json

from benchmarks.common import Benchmark, Suite

from .language import AsmLanguage
from .learnr_bridge import PROGRAMS_PATH
from .term import Const, Expr, Var, from_sexpr

x, y, z, w = (Var(name) for name in ("x", "y", "z", "w"))


def B(op: str, a: Expr, b: Expr) -> Expr:
    return Expr(op, (a, b))


def _learnr_programs() -> tuple[Benchmark, ...]:
    """Learn-R's own straight-line test programs, compiled by its frontend."""
    if not PROGRAMS_PATH.exists():
        return ()
    programs = json.loads(PROGRAMS_PATH.read_text())
    return tuple(
        Benchmark(
            f"learnr_{name}",
            from_sexpr(program["terms"]["final"]),
            "easy",
            f"Learn-R {name}.txt compiled by Learn-R's frontend (variable 'final').",
        )
        for name, program in sorted(programs.items())
        if program["straight_line"] and "final" in program["terms"]
    )


SUITE = Suite(
    "asm",
    AsmLanguage(),
    (
        *_learnr_programs(),
        Benchmark(
            "copy_chain",
            B("add", B("mul", B("mul", x, Const(1)), Const(2)), Const(0)),
            "easy",
            "Compiler-style chain: drop identities, then strength-reduce the doubling.",
        ),
        Benchmark(
            "double_negation",
            B("xor", Expr("not", (Expr("not", (Expr("neg", (Expr("neg", (x,)),)),)),)), Const(0)),
            "easy",
            "Nested complements and negations around an empty XOR.",
        ),
        Benchmark(
            "self_cancel",
            B("add", B("mul", B("sub", B("add", x, y), B("add", x, y)), z), w),
            "medium",
            "A difference of equal sums zeroes a product that hides w.",
        ),
        Benchmark(
            "shift_fold",
            B("mul", B("mul", x, Const(2)), Const(4)),
            "hard",
            "Reassociate, fold 2*4, then strength-reduce x*8 to a shift.",
        ),
        Benchmark(
            "load_twice",
            B("add", Expr("load", (B("add", x, Const(0)),)), Expr("load", (x,))),
            "hard",
            "Normalise an address so two loads match, then double with a shift.",
        ),
    ),
)
