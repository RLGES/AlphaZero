"""Z3 proof that a pattern rule holds for every 32-bit input.

Encodes ``term.apply_op`` exactly: unsigned wrap-around, division and modulo by
zero give 0, shift counts masked to 5 bits, ``load`` an uninterpreted function
and ``phi(a, b)`` equal to ``a`` when ``a == b`` (otherwise uninterpreted).
``verify_rule`` in ``patterns`` only tests random inputs; this is a proof.
"""

from __future__ import annotations

from .patterns import Pattern, parse_pattern, variables
from .term import MASK


def z3_available() -> bool:
    try:
        import z3  # noqa: F401
    except ImportError:
        return False
    return True


def _encode(pattern: Pattern, env: dict, z3, funcs: dict):
    if isinstance(pattern, str):
        return env[pattern[1:]]
    if isinstance(pattern, int):
        return z3.BitVecVal(pattern & MASK, 32)
    op, *args = pattern
    values = [_encode(arg, env, z3, funcs) for arg in args]
    zero = z3.BitVecVal(0, 32)
    if op == "neg":
        return -values[0]
    if op == "not":
        return ~values[0]
    if op == "load":
        return funcs["load"](values[0])
    a, b = values
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return z3.If(b == zero, zero, z3.UDiv(a, b))
    if op == "mod":
        return z3.If(b == zero, zero, z3.URem(a, b))
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    if op == "shl":
        return a << (b & 31)
    if op == "shr":
        return z3.LShR(a, b & 31)
    if op == "phi":
        return z3.If(a == b, a, funcs["phi"](a, b))
    raise ValueError(f"unknown operator {op!r}")


def prove_rule(lhs: str, rhs: str, timeout_ms: int = 10_000) -> tuple[str, str | None]:
    """("proved", None), ("counterexample", model text) or ("unknown", reason)."""
    import z3

    left, right = parse_pattern(lhs), parse_pattern(rhs)
    if not variables(right) <= variables(left):
        return "counterexample", "right side uses unbound variables"
    env = {name: z3.BitVec(name, 32) for name in sorted(variables(left))}
    bv = z3.BitVecSort(32)
    funcs = {"load": z3.Function("load", bv, bv), "phi": z3.Function("phi", bv, bv, bv)}
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(_encode(left, env, z3, funcs) != _encode(right, env, z3, funcs))
    result = solver.check()
    if result == z3.unsat:
        return "proved", None
    if result == z3.sat:
        return "counterexample", str(solver.model())
    return "unknown", solver.reason_unknown()
