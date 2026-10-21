"""Random 8-bit training terms with planted redundancy."""

from random import Random

from .term import Add, And, Const, Expr, Not, Or, Var, Xor

VARIABLES = tuple(Var(name) for name in ("x", "y", "z"))


def _leaf(rng: Random) -> Expr:
    if rng.random() < 0.65:
        return rng.choice(VARIABLES)
    return Const(rng.choice((0, 255, 1, 15, 240, 128)))


def _plant(rng: Random, t: Expr, depth: int) -> Expr:
    kind = rng.randrange(6)
    if kind == 0:  # empty XOR / OR
        return Xor(t, Const(0)) if rng.random() < 0.5 else Or(Const(0), t)
    if kind == 1:  # all-ones mask
        return And(t, Const(255))
    if kind == 2:  # double complement
        return Not(Not(t))
    if kind == 3:  # self-cancelling XOR behind association
        other = random_term(rng, depth - 1)
        return Xor(Xor(t, other), other)
    if kind == 4:  # wraparound constant folding
        return Add(Add(t, Const(rng.randint(200, 255))), Const(rng.randint(1, 60)))
    return t


def random_term(rng: Random, depth: int) -> Expr:
    if depth <= 0:
        return _leaf(rng)
    roll = rng.random()
    if roll < 0.1:
        t = Not(random_term(rng, depth - 1))
    else:
        make = rng.choice((And, Or, Xor, Add))
        t = make(random_term(rng, depth - 1), random_term(rng, depth - 1))
    return _plant(rng, t, depth) if rng.random() < 0.5 else t
