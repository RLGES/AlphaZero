"""Random arithmetic training terms with planted redundancy."""

from random import Random

from .term import Add, Const, Expr, Mul, Var

VARIABLES = tuple(Var(name) for name in ("x", "y", "z", "w"))


def _leaf(rng: Random) -> Expr:
    if rng.random() < 0.6:
        return rng.choice(VARIABLES)
    return Const(rng.choice((0, 1, 1, 2, 3, -1, 5)))


def _plant(rng: Random, t: Expr, depth: int) -> Expr:
    kind = rng.randrange(6)
    if kind == 0:
        return Add(t, Const(0)) if rng.random() < 0.5 else Add(Const(0), t)
    if kind == 1:
        return Mul(t, Const(1)) if rng.random() < 0.5 else Mul(Const(1), t)
    if kind == 2:
        return Add(t, Mul(random_term(rng, depth - 1), Const(0)))
    if kind == 3:  # shared factor
        shared = _leaf(rng)
        return Add(Mul(shared, t), Mul(random_term(rng, depth - 1), shared))
    if kind == 4:  # foldable constants hidden behind association
        return Add(Add(t, Const(rng.randint(1, 4))), Const(rng.randint(1, 4)))
    return t


def random_term(rng: Random, depth: int) -> Expr:
    if depth <= 0:
        return _leaf(rng)
    make = Add if rng.random() < 0.5 else Mul
    t = make(random_term(rng, depth - 1), random_term(rng, depth - 1))
    return _plant(rng, t, depth) if rng.random() < 0.5 else t
