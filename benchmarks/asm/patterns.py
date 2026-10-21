"""Rewrite rules as data: s-expression patterns over the asm language.

``(add ?x 0)`` matches an addition whose right operand is the constant 0 and
binds ``x``; a variable used twice (``(sub ?x ?x)``) must match equal subterms.
The same representation serves Learn-R's built-in rules and imported learned
rules, and ``verify_rule`` checks soundness by evaluating both sides on random
and edge-case 32-bit inputs.
"""

from __future__ import annotations

from random import Random
from typing import Any

from benchmarks.common import TreeRule

from .term import MASK, Const, Expr, apply_op, parse_tokens, tokenize

Pattern = Any  # "?x" | int | (op, Pattern, ...)


def parse_pattern(text: str) -> Pattern:
    tree, end = parse_tokens(tokenize(text))
    if end != len(tokenize(text)):
        raise ValueError(f"Trailing input in pattern {text!r}")
    return _normalize(tree)


def _normalize(tree) -> Pattern:
    if isinstance(tree, tuple):
        return (tree[0], *(_normalize(arg) for arg in tree[1:]))
    if tree.startswith("?"):
        return tree
    return int(tree)


def variables(pattern: Pattern) -> set[str]:
    if isinstance(pattern, str):
        return {pattern[1:]}
    if isinstance(pattern, tuple):
        return set().union(*(variables(arg) for arg in pattern[1:]))
    return set()


def match(pattern: Pattern, term: Expr, bindings: dict | None = None) -> dict | None:
    bindings = dict(bindings or {})
    if isinstance(pattern, str):
        name = pattern[1:]
        if name in bindings:
            return bindings if bindings[name] == term else None
        bindings[name] = term
        return bindings
    if isinstance(pattern, int):
        return bindings if term == Const(pattern) else None
    op, *args = pattern
    if term.op != op or len(term.args) != len(args):
        return None
    for sub_pattern, sub_term in zip(args, term.args):
        bindings = match(sub_pattern, sub_term, bindings)
        if bindings is None:
            return None
    return bindings


def build(pattern: Pattern, bindings: dict) -> Expr:
    if isinstance(pattern, str):
        return bindings[pattern[1:]]
    if isinstance(pattern, int):
        return Const(pattern)
    op, *args = pattern
    return Expr(op, tuple(build(arg, bindings) for arg in args))


def _evaluate(pattern: Pattern, env: dict[str, int]) -> int:
    if isinstance(pattern, str):
        return env[pattern[1:]]
    if isinstance(pattern, int):
        return pattern & MASK
    op, *args = pattern
    return apply_op(op, [_evaluate(arg, env) for arg in args])


EDGE_VALUES = (0, 1, 2, 3, 31, 32, MASK, MASK - 1, 1 << 31, (1 << 31) - 1)


def verify_rule(lhs: str, rhs: str, rng: Random, trials: int = 2000) -> bool:
    """True when both sides agree on edge-case and random 32-bit assignments."""
    left, right = parse_pattern(lhs), parse_pattern(rhs)
    names = sorted(variables(left))
    if not variables(right) <= set(names):
        return False
    for trial in range(trials):
        if trial < 64:
            env = {name: rng.choice(EDGE_VALUES) for name in names}
        else:
            env = {name: rng.getrandbits(32) for name in names}
        if _evaluate(left, env) != _evaluate(right, env):
            return False
    return True


def pattern_rule(name: str, lhs: str, rhs: str) -> TreeRule[Expr]:
    left, right = parse_pattern(lhs), parse_pattern(rhs)

    def transform(term: Expr) -> Expr | None:
        bindings = match(left, term)
        return None if bindings is None else build(right, bindings)

    return TreeRule(name, transform)
