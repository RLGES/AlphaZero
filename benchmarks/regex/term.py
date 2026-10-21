"""Classical regular languages (set union), without captures or ordered alternation."""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    symbol: str | None = None

    def __post_init__(self) -> None:
        arity = {"empty": 0, "eps": 0, "lit": 0, "star": 1, "alt": 2, "seq": 2}
        valid = self.op in arity and len(self.args) == arity[self.op]
        if self.op == "lit":
            valid = valid and isinstance(self.symbol, str) and len(self.symbol) == 1
        else:
            valid = valid and self.symbol is None
        if not valid:
            raise ValueError("Invalid regular expression")

    def __str__(self) -> str:
        if self.op in ("empty", "eps"):
            return self.op
        if self.op == "lit":
            return repr(self.symbol)
        if self.op == "star":
            return f"({self.args[0]})*"
        return f"({self.args[0]} {'|' if self.op == 'alt' else '.'} {self.args[1]})"


def Empty() -> Expr:
    return Expr("empty")


def Eps() -> Expr:
    return Expr("eps")


def Lit(symbol: str) -> Expr:
    return Expr("lit", symbol=symbol)


def Alt(a: Expr, b: Expr) -> Expr:
    return Expr("alt", (a, b))


def Seq(a: Expr, b: Expr) -> Expr:
    return Expr("seq", (a, b))


def Star(a: Expr) -> Expr:
    return Expr("star", (a,))


def matches(term: Expr, text: str) -> bool:
    @lru_cache(maxsize=None)
    def ends(t: Expr, start: int) -> frozenset[int]:
        if t.op == "empty":
            return frozenset()
        if t.op == "eps":
            return frozenset((start,))
        if t.op == "lit":
            return frozenset((start + 1,)) if text[start : start + 1] == t.symbol else frozenset()
        if t.op == "alt":
            return ends(t.args[0], start) | ends(t.args[1], start)
        if t.op == "seq":
            return frozenset(j for i in ends(t.args[0], start) for j in ends(t.args[1], i))
        # Nullable bodies and nested stars terminate: each string position is visited once.
        reached, pending = {start}, [start]
        while pending:
            for end in ends(t.args[0], pending.pop()):
                if end not in reached:
                    reached.add(end)
                    pending.append(end)
        return frozenset(reached)

    return len(text) in ends(term, 0)
