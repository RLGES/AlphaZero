from __future__ import annotations

from egglog import BigInt, EGraph, String, birewrite, rewrite, ruleset, vars_
from egglog import Expr as EggExpr
from egglog.deconstruct import get_callable_args

from .term import Add, Const, Expr, Mul, Var


class ArithmeticExpr(EggExpr):
    def __init__(self, value: BigInt) -> None: ...

    @classmethod
    def var(cls, name: String) -> ArithmeticExpr: ...

    def __add__(self, other: ArithmeticExpr) -> ArithmeticExpr: ...
    def __mul__(self, other: ArithmeticExpr) -> ArithmeticExpr: ...


class ArithmeticAdapter:
    def encode(self, term: Expr) -> ArithmeticExpr:
        if term.op == "const":
            return ArithmeticExpr(BigInt.from_string(str(term.value)))
        if term.op == "var":
            return ArithmeticExpr.var(str(term.value))
        a, b = (self.encode(child) for child in term.args)
        return a + b if term.op == "add" else a * b

    def decode(self, term: ArithmeticExpr) -> Expr:
        if (args := get_callable_args(term, ArithmeticExpr)) is not None:
            return Const(int(args[0]))
        if (args := get_callable_args(term, ArithmeticExpr.var)) is not None:
            return Var(str(args[0].value))
        if (args := get_callable_args(term, ArithmeticExpr.__add__)) is not None:
            return Add(*(self.decode(arg) for arg in args))
        if (args := get_callable_args(term, ArithmeticExpr.__mul__)) is not None:
            return Mul(*(self.decode(arg) for arg in args))
        raise ValueError(f"Cannot decode arithmetic egglog expression: {term}")

    def ruleset(self):
        a, b, c = vars_("a b c", ArithmeticExpr)
        i, j = vars_("i j", BigInt)
        zero, one = self.encode(Const(0)), self.encode(Const(1))
        return ruleset(
            rewrite(a + b).to(b + a),
            rewrite(a * b).to(b * a),
            birewrite(a + (b + c)).to((a + b) + c),
            birewrite(a * (b * c)).to((a * b) * c),
            birewrite(a * (b + c)).to(a * b + a * c),
            rewrite(a + zero).to(a),
            rewrite(a * one).to(a),
            rewrite(a * zero).to(zero),
            rewrite(ArithmeticExpr(i) + ArithmeticExpr(j)).to(ArithmeticExpr(i + j)),
            rewrite(ArithmeticExpr(i) * ArithmeticExpr(j)).to(ArithmeticExpr(i * j)),
            name="arithmetic",
        )

    def cost_model(self, egraph: EGraph, expr: EggExpr, child_costs: list[int]) -> int:
        # IR nodes cost one; egglog's primitive payload representation costs zero.
        return sum(child_costs) + int(isinstance(expr, ArithmeticExpr))
