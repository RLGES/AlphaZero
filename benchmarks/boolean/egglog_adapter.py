from __future__ import annotations

from egglog import Bool as EggBool
from egglog import EGraph, String, birewrite, rewrite, ruleset, vars_
from egglog import Expr as EggExpr
from egglog.deconstruct import get_callable_args

from .term import And, Bool, Expr, Not, Or, Var


class BooleanExpr(EggExpr):
    def __init__(self, value: EggBool) -> None: ...

    @classmethod
    def var(cls, name: String) -> BooleanExpr: ...

    def __and__(self, other: BooleanExpr) -> BooleanExpr: ...
    def __or__(self, other: BooleanExpr) -> BooleanExpr: ...
    def __invert__(self) -> BooleanExpr: ...


class BooleanAdapter:
    def encode(self, term: Expr) -> BooleanExpr:
        if term.op == "bool":
            return BooleanExpr(bool(term.value))
        if term.op == "var":
            return BooleanExpr.var(str(term.value))
        if term.op == "not":
            return ~self.encode(term.args[0])
        a, b = (self.encode(child) for child in term.args)
        return a & b if term.op == "and" else a | b

    def decode(self, term: BooleanExpr) -> Expr:
        if (args := get_callable_args(term, BooleanExpr)) is not None:
            return Bool(bool(args[0]))
        if (args := get_callable_args(term, BooleanExpr.var)) is not None:
            return Var(str(args[0].value))
        for fn, make in (
            (BooleanExpr.__and__, And),
            (BooleanExpr.__or__, Or),
            (BooleanExpr.__invert__, Not),
        ):
            if (args := get_callable_args(term, fn)) is not None:
                return make(*(self.decode(arg) for arg in args))
        raise ValueError(f"Cannot decode Boolean egglog expression: {term}")

    def ruleset(self):
        a, b, c = vars_("a b c", BooleanExpr)
        i, j = vars_("i j", EggBool)
        true, false = self.encode(Bool(True)), self.encode(Bool(False))
        return ruleset(
            rewrite(a & b).to(b & a),
            rewrite(a | b).to(b | a),
            birewrite(a & (b & c)).to((a & b) & c),
            birewrite(a | (b | c)).to((a | b) | c),
            birewrite(a & (b | c)).to((a & b) | (a & c)),
            birewrite(a | (b & c)).to((a | b) & (a | c)),
            birewrite(~(a & b)).to(~a | ~b),
            birewrite(~(a | b)).to(~a & ~b),
            rewrite(~~a).to(a),
            rewrite(a & true).to(a),
            rewrite(a | false).to(a),
            rewrite(a & false).to(false),
            rewrite(a | true).to(true),
            rewrite(a & a).to(a),
            rewrite(a | a).to(a),
            rewrite(a & ~a).to(false),
            rewrite(a | ~a).to(true),
            rewrite(a | (a & b)).to(a),
            rewrite(a & (a | b)).to(a),
            rewrite(~BooleanExpr(i)).to(BooleanExpr(~i)),
            rewrite(BooleanExpr(i) & BooleanExpr(j)).to(BooleanExpr(i & j)),
            rewrite(BooleanExpr(i) | BooleanExpr(j)).to(BooleanExpr(i | j)),
            name="boolean",
        )

    def cost_model(self, egraph: EGraph, expr: EggExpr, child_costs: list[int]) -> int:
        return sum(child_costs) + int(isinstance(expr, BooleanExpr))
