"""Exact commutative integer arithmetic: addition and multiplication only."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value: int | str | None = None

    def __post_init__(self) -> None:
        if self.op == "const":
            valid = type(self.value) is int and not self.args
        elif self.op == "var":
            valid = isinstance(self.value, str) and not self.args
        else:
            valid = self.op in ("add", "mul") and len(self.args) == 2 and self.value is None
        if not valid:
            raise ValueError("Invalid arithmetic term")

    def __str__(self) -> str:
        if not self.args:
            return str(self.value)
        return f"({self.args[0]} {'+' if self.op == 'add' else '*'} {self.args[1]})"


def Const(value: int) -> Expr:
    return Expr("const", value=value)


def Var(name: str) -> Expr:
    return Expr("var", value=name)


def Add(a: Expr, b: Expr) -> Expr:
    return Expr("add", (a, b))


def Mul(a: Expr, b: Expr) -> Expr:
    return Expr("mul", (a, b))


def evaluate(term: Expr, env: dict[str, int]) -> int:
    if term.op == "const":
        return int(term.value)
    if term.op == "var":
        return env[str(term.value)]
    a, b = (evaluate(child, env) for child in term.args)
    return a + b if term.op == "add" else a * b
