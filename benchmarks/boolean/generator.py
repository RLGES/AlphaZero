"""Random Boolean training terms with planted redundancy."""

from random import Random

from .term import And, Bool, Expr, Not, Or, Var

VARIABLES = tuple(Var(name) for name in ("p", "q", "r", "s"))


def _leaf(rng: Random) -> Expr:
    return rng.choice(VARIABLES) if rng.random() < 0.8 else Bool(rng.random() < 0.5)


def _plant(rng: Random, t: Expr, depth: int) -> Expr:
    kind = rng.randrange(6)
    if kind == 0:  # identity element
        return And(t, Bool(True)) if rng.random() < 0.5 else Or(Bool(False), t)
    if kind == 1:  # double negation
        return Not(Not(t))
    if kind == 2:  # absorption
        return Or(t, And(t, random_term(rng, depth - 1)))
    if kind == 3:  # De Morgan pair that cancels into a factor
        other = random_term(rng, depth - 1)
        return Not(Or(Not(t), Not(other)))
    if kind == 4:  # shared factor
        shared = _leaf(rng)
        return Or(And(shared, t), And(random_term(rng, depth - 1), shared))
    return t


def random_term(rng: Random, depth: int) -> Expr:
    if depth <= 0:
        return _leaf(rng)
    roll = rng.random()
    if roll < 0.15:
        t = Not(random_term(rng, depth - 1))
    else:
        make = And if roll < 0.575 else Or
        t = make(random_term(rng, depth - 1), random_term(rng, depth - 1))
    return _plant(rng, t, depth) if rng.random() < 0.5 else t
