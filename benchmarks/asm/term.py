"""Learn-R's Asm expression language with 32-bit machine semantics.

Mirrors ``Learn-R/egraph_bridge/egg_egraph.py``'s ``Asm`` type. Values are
unsigned 32-bit words; division and modulo by zero give 0, and shift counts are
masked to 5 bits as on x86. ``load`` and two-input ``phi`` are uninterpreted
(deterministic hashes), except that ``phi(v, v) = v``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

MASK = 0xFFFF_FFFF
BINARY = ("add", "sub", "mul", "div", "mod", "and", "or", "xor", "shl", "shr", "phi")
UNARY = ("neg", "not", "load")
SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
    "and": "&",
    "or": "|",
    "xor": "^",
    "shl": "<<",
    "shr": ">>",
}


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value: int | str | None = None

    def __post_init__(self) -> None:
        if self.op == "const":
            valid = type(self.value) is int and not self.args
            if valid:
                object.__setattr__(self, "value", self.value & MASK)
        elif self.op == "var":
            valid = isinstance(self.value, str) and bool(self.value) and not self.args
        elif self.op in BINARY:
            valid = len(self.args) == 2 and self.value is None
        elif self.op in UNARY:
            valid = len(self.args) == 1 and self.value is None
        else:
            valid = False
        if not valid:
            raise ValueError(f"Invalid asm term: {self.op} {self.args} {self.value}")

    @property
    def signed(self) -> int:
        value = int(self.value)
        return value - (1 << 32) if value >> 31 else value

    def __str__(self) -> str:
        if self.op == "const":
            return str(self.signed)
        if self.op == "var":
            return str(self.value)
        if self.op in SYMBOLS:
            return f"({self.args[0]} {SYMBOLS[self.op]} {self.args[1]})"
        if self.op == "neg":
            return f"-{self.args[0]}"
        if self.op == "not":
            return f"~{self.args[0]}"
        return f"{self.op}({', '.join(str(a) for a in self.args)})"


def Const(value: int) -> Expr:
    return Expr("const", value=value)


def Var(name: str) -> Expr:
    return Expr("var", value=name)


def _opaque(*parts: object) -> int:
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def apply_op(op: str, values: list[int]) -> int:
    if op in UNARY:
        (a,) = values
        if op == "neg":
            return -a & MASK
        if op == "not":
            return ~a & MASK
        return _opaque("load", a)
    a, b = values
    if op == "add":
        return (a + b) & MASK
    if op == "sub":
        return (a - b) & MASK
    if op == "mul":
        return (a * b) & MASK
    if op == "div":
        return a // b if b else 0
    if op == "mod":
        return a % b if b else 0
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    if op == "shl":
        return (a << (b & 31)) & MASK
    if op == "shr":
        return a >> (b & 31)
    return a if a == b else _opaque("phi", a, b)


def evaluate(term: Expr, env: dict[str, int]) -> int:
    if term.op == "const":
        return int(term.value)
    if term.op == "var":
        return env[str(term.value)] & MASK
    return apply_op(term.op, [evaluate(arg, env) for arg in term.args])


def to_sexpr(term: Expr) -> str:
    if term.op == "const":
        return str(term.signed)
    if term.op == "var":
        return str(term.value)
    return f"({term.op} {' '.join(to_sexpr(arg) for arg in term.args)})"


def tokenize(text: str) -> list[str]:
    return text.replace("(", " ( ").replace(")", " ) ").split()


def parse_tokens(tokens: list[str], position: int = 0):
    """Parse one s-expression into nested tuples/atoms; returns (tree, next position)."""
    token = tokens[position]
    if token != "(":
        return token, position + 1
    items = []
    position += 1
    while tokens[position] != ")":
        item, position = parse_tokens(tokens, position)
        items.append(item)
    return tuple(items), position + 1


def from_sexpr(text: str) -> Expr:
    tree, end = parse_tokens(tokenize(text))
    if end != len(tokenize(text)):
        raise ValueError(f"Trailing input in {text!r}")
    return _from_tree(tree)


def _from_tree(tree) -> Expr:
    if isinstance(tree, tuple):
        op, *args = tree
        return Expr(op, tuple(_from_tree(arg) for arg in args))
    try:
        return Const(int(tree))
    except ValueError:
        return Var(tree)
