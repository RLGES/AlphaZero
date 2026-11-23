"""egglog adapter for the asm suite: bounded equality saturation as a baseline.

Every rewrite rule the search engines use becomes an egglog rewrite: Learn-R's
built-in patterns plus the learned rules selected by ``RLGES_ASM_LEARNED``.
Constant folding follows the 32-bit semantics of ``term.apply_op`` (``load``
and ``phi`` stay uninterpreted), and extraction uses the suite's cost model.
"""

from __future__ import annotations

import json
import os

from egglog import EGraph, String, i64, ne, rewrite, ruleset, vars_
from egglog import Expr as EggExpr
from egglog.deconstruct import get_callable_args

from .language import OP_COST
from .learned import learned_path
from .patterns import parse_pattern, variables
from .rules import PATTERN_RULES
from .term import BINARY, MASK, UNARY, Const, Expr, Var


class AsmExpr(EggExpr):
    def __init__(self, value: i64) -> None: ...

    @classmethod
    def var(cls, name: String) -> AsmExpr: ...

    def add(self, other: AsmExpr) -> AsmExpr: ...
    def sub(self, other: AsmExpr) -> AsmExpr: ...
    def mul(self, other: AsmExpr) -> AsmExpr: ...
    def div(self, other: AsmExpr) -> AsmExpr: ...
    def mod(self, other: AsmExpr) -> AsmExpr: ...
    def and_(self, other: AsmExpr) -> AsmExpr: ...
    def or_(self, other: AsmExpr) -> AsmExpr: ...
    def xor(self, other: AsmExpr) -> AsmExpr: ...
    def shl(self, other: AsmExpr) -> AsmExpr: ...
    def shr(self, other: AsmExpr) -> AsmExpr: ...
    def phi(self, other: AsmExpr) -> AsmExpr: ...
    def neg(self) -> AsmExpr: ...
    def not_(self) -> AsmExpr: ...
    def load(self) -> AsmExpr: ...


# asm operator -> AsmExpr method name ("and", "or" and "not" are Python keywords).
METHOD = {op: op + "_" if op in ("and", "or", "not") else op for op in BINARY + UNARY}


def _const(value: int) -> AsmExpr:
    return AsmExpr(i64(value & MASK))


def _apply(op: str, args: list[AsmExpr]) -> AsmExpr:
    first, *rest = args
    return getattr(first, METHOD[op])(*rest)


def learned_patterns() -> list[tuple[str, str]]:
    """(lhs, rhs) of the learned rules the search engines load, if any."""
    return [(lhs, rhs) for _, lhs, rhs in learned_named()]


def learned_named() -> list[tuple[str, str, str]]:
    """(name, lhs, rhs) of the learned rules, named as in ``AsmLanguage.rules``."""
    path = learned_path()
    if path is None or not path.exists():
        return []
    return [(r["name"], r["lhs"], r["rhs"]) for r in json.loads(path.read_text())["accepted"]]


class AsmAdapter:
    def encode(self, term: Expr) -> AsmExpr:
        if term.op == "const":
            return _const(int(term.value))
        if term.op == "var":
            return AsmExpr.var(str(term.value))
        return _apply(term.op, [self.encode(arg) for arg in term.args])

    def decode(self, term: AsmExpr) -> Expr:
        if (args := get_callable_args(term, AsmExpr)) is not None:
            return Const(int(args[0].value))
        if (args := get_callable_args(term, AsmExpr.var)) is not None:
            return Var(str(args[0].value))
        for op, method in METHOD.items():
            if (args := get_callable_args(term, getattr(AsmExpr, method))) is not None:
                return Expr(op, tuple(self.decode(arg) for arg in args))
        raise ValueError(f"Cannot decode asm egglog expression: {term}")

    def _pattern(self, pattern, slots: dict[str, AsmExpr]) -> AsmExpr:
        if isinstance(pattern, str):
            return slots[pattern[1:]]
        if isinstance(pattern, int):
            return _const(pattern)
        op, *args = pattern
        return _apply(op, [self._pattern(arg, slots) for arg in args])

    def rewrite_rule(self, lhs: str, rhs: str):
        left, right = parse_pattern(lhs), parse_pattern(rhs)
        names = sorted(variables(left))
        slots = dict(zip(names, vars_(" ".join(names), AsmExpr))) if names else {}
        return rewrite(self._pattern(left, slots)).to(self._pattern(right, slots))

    def fold_rules(self) -> list:
        # Payloads are unsigned 32-bit values in i64; results are masked back to 32 bits.
        i, j = vars_("i j", i64)
        a, b = AsmExpr(i), AsmExpr(j)
        m = i64(MASK)
        # 32x32-bit products overflow i64, so multiply by the low and high 16 bits apart.
        product = ((i & i64(0xFFFF)) * j + ((((i >> i64(16)) * j) & i64(0xFFFF)) << i64(16))) & m
        return [
            rewrite(a.add(b)).to(AsmExpr((i + j) & m)),
            rewrite(a.sub(b)).to(AsmExpr((i - j) & m)),
            rewrite(a.mul(b)).to(AsmExpr(product)),
            rewrite(a.and_(b)).to(AsmExpr(i & j)),
            rewrite(a.or_(b)).to(AsmExpr(i | j)),
            rewrite(a.xor(b)).to(AsmExpr(i ^ j)),
            rewrite(a.shl(b)).to(AsmExpr((i << (j & i64(31))) & m)),
            rewrite(a.shr(b)).to(AsmExpr(i >> (j & i64(31)))),
            rewrite(a.div(b)).to(AsmExpr(i / j), ne(j).to(i64(0))),
            rewrite(a.mod(b)).to(AsmExpr(i % j), ne(j).to(i64(0))),
            rewrite(a.div(_const(0))).to(_const(0)),
            rewrite(a.mod(_const(0))).to(_const(0)),
            rewrite(a.neg()).to(AsmExpr((i64(0) - i) & m)),
            rewrite(a.not_()).to(AsmExpr(i ^ m)),
        ]

    def ruleset(self):
        named = list(PATTERN_RULES) + learned_named()
        # RLGES_EGGLOG_RULES (comma-separated names) keeps only those pattern rules;
        # constant folding always stays. Used by the AZ-chosen-rules hybrid.
        keep = os.environ.get("RLGES_EGGLOG_RULES")
        if keep is not None:
            wanted = set(keep.split(",")) if keep else set()
            named = [rule for rule in named if rule[0] in wanted]
        patterns = [(lhs, rhs) for _, lhs, rhs in named]
        return ruleset(
            *self.fold_rules(), *(self.rewrite_rule(lhs, rhs) for lhs, rhs in patterns), name="asm"
        )

    def cost_model(self, egraph: EGraph, expr: EggExpr, child_costs: list[int]) -> int:
        # Same costs as AsmLanguage.cost; egglog's primitive payloads cost zero.
        if not isinstance(expr, AsmExpr):
            return sum(child_costs)
        for op, method in METHOD.items():
            if get_callable_args(expr, getattr(AsmExpr, method)) is not None:
                return sum(child_costs) + OP_COST.get(op, 1)
        return sum(child_costs) + 1
