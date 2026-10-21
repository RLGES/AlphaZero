"""Unsigned fixed-width words; arithmetic wraps modulo 2**width."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Expr:
    op: str
    width: int
    args: tuple["Expr", ...] = ()
    value: int | str | None = None

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 1:
            raise ValueError("Bit width must be positive")
        arity = {"not": 1, "and": 2, "or": 2, "xor": 2, "add": 2}
        valid = (
            (self.op == "const" and type(self.value) is int and not self.args)
            or (self.op == "var" and isinstance(self.value, str) and not self.args)
            or (self.op in arity and len(self.args) == arity[self.op] and self.value is None)
        )
        if not valid or any(arg.width != self.width for arg in self.args):
            raise ValueError("Invalid bitvector term or mismatched widths")
        if self.op == "const":
            object.__setattr__(self, "value", self.value & self.mask)

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1

    def __str__(self) -> str:
        if not self.args:
            return f"{self.value}:{self.width}"
        if self.op == "not":
            return f"~{self.args[0]}"
        symbol = {"and": "&", "or": "|", "xor": "^", "add": "+"}[self.op]
        return f"({self.args[0]} {symbol} {self.args[1]})"


def Const(value: int, width: int = 8) -> Expr:
    return Expr("const", width, value=value)


def Var(name: str, width: int = 8) -> Expr:
    return Expr("var", width, value=name)


def Not(a: Expr) -> Expr:
    return Expr("not", a.width, (a,))


def And(a: Expr, b: Expr) -> Expr:
    return Expr("and", a.width, (a, b))


def Or(a: Expr, b: Expr) -> Expr:
    return Expr("or", a.width, (a, b))


def Xor(a: Expr, b: Expr) -> Expr:
    return Expr("xor", a.width, (a, b))


def Add(a: Expr, b: Expr) -> Expr:
    return Expr("add", a.width, (a, b))


def evaluate(t: Expr, env: dict[str, int]) -> int:
    if t.op == "const":
        return int(t.value)
    if t.op == "var":
        return env[str(t.value)] & t.mask
    values = [evaluate(child, env) for child in t.args]
    if t.op == "not":
        return ~values[0] & t.mask
    a, b = values
    result = {
        "and": lambda: a & b,
        "or": lambda: a | b,
        "xor": lambda: a ^ b,
        "add": lambda: a + b,
    }[t.op]()
    return result & t.mask
