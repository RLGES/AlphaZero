"""Training terms for the asm suite.

Most terms come from Learn-R-compiled programs (``corpus/<split>.jsonl``, made by
``learnr_bridge``); the rest are synthetic assembly-level terms covering the
bitwise, shift, division and memory operations that Learn-R's source language
cannot express but its rules (and strength reduction) produce. ``split="test"``
draws from the held-out corpus file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from random import Random

from .learnr_bridge import CORPUS_DIR
from .term import Const, Expr, Var, from_sexpr

REGISTERS = tuple(Var(f"r{i}") for i in range(4))
CONSTANTS = (0, 1, 2, 3, 4, 8, 16, -1, 255)
BINARY = ("add", "sub", "mul", "div", "mod", "and", "or", "xor", "shl", "shr")
CORPUS_SHARE = 0.6


@lru_cache(maxsize=None)
def corpus(split: str) -> tuple[Expr, ...]:
    """Straight-line terms plus basic-block terms from if/while programs (``cf_``)."""
    terms: list[Expr] = []
    for name in (f"{split}.jsonl", f"cf_{split}.jsonl"):
        path = CORPUS_DIR / name
        if path.exists():
            terms += (from_sexpr(json.loads(line)["sexpr"]) for line in path.open())
    return tuple(terms)


def _leaf(rng: Random) -> Expr:
    return rng.choice(REGISTERS) if rng.random() < 0.65 else Const(rng.choice(CONSTANTS))


def _plant(rng: Random, t: Expr, depth: int) -> Expr:
    kind = rng.randrange(9)
    if kind == 0:
        return Expr(rng.choice(("xor", "or", "add", "shl", "shr")), (t, Const(0)))
    if kind == 1:
        return Expr(rng.choice(("mul", "div")), (t, Const(rng.choice((2, 4, 8, 16)))))
    if kind == 2:
        return Expr(rng.choice(("neg", "not")), (Expr(rng.choice(("neg", "not")), (t,)),))
    if kind == 3:
        return Expr(rng.choice(("xor", "sub", "and", "or")), (t, t))
    if kind == 4:
        return Expr("add", (t, t))
    if kind == 5:
        return Expr("load", (Expr("add", (t, Const(0))),))
    if kind == 6:
        return Expr("add", (Expr("mul", (random_synthetic(rng, depth - 1), Const(0))), t))
    if kind == 7:
        return Expr("mul", (Expr("mul", (t, Const(2))), Const(rng.choice((2, 4)))))
    return t


def random_synthetic(rng: Random, depth: int) -> Expr:
    if depth <= 0:
        return _leaf(rng)
    roll = rng.random()
    if roll < 0.1:
        t = Expr(rng.choice(("neg", "not", "load")), (random_synthetic(rng, depth - 1),))
    else:
        op = rng.choice(BINARY)
        t = Expr(op, (random_synthetic(rng, depth - 1), random_synthetic(rng, depth - 1)))
    return _plant(rng, t, depth) if rng.random() < 0.5 else t


def random_term(rng: Random, depth: int, split: str = "train") -> Expr:
    terms = corpus(split)
    if terms and rng.random() < CORPUS_SHARE:
        return rng.choice(terms)
    return random_synthetic(rng, depth)
