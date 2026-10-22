"""Hardness filter: keep terms where a quick search falls short of a strong one.

Most randomly planted redundancy is a one-step greedy win, so untrained search
already solves it and training has nothing to learn. A term is *hard* when
greedy descent or a small MCTS budget (either one) ends above the cost a large
MCTS budget reaches, i.e. some non-improving step (commute, reassociate) is
needed first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor

from .term import Expr

QUICK_BUDGET = 10
STRONG_BUDGET = 150


def greedy_cost(language, term: Expr, max_steps: int = 64) -> float:
    """Apply the single most cost-reducing action until none reduces the cost."""
    cost = language.cost(term)
    for _ in range(max_steps):
        best = None
        for action in language.applicable(term):
            candidate = language.apply(term, action)
            candidate_cost = language.cost(candidate)
            if candidate_cost < cost and (best is None or candidate_cost < best[0]):
                best = (candidate_cost, candidate)
        if best is None:
            break
        cost, term = best
    return cost


def hardness(term: Expr) -> dict:
    """Costs reached by greedy, MCTS@QUICK and MCTS@STRONG on one term."""
    from src.api import rewrite_detailed
    from src.engines.mcts import MCTSConfig

    from .language import AsmLanguage

    language = AsmLanguage()

    def mcts(budget: int) -> float:
        return rewrite_detailed(language, term, MCTSConfig(iterations=budget, seed=0)).final_cost

    strong = mcts(STRONG_BUDGET)
    return {
        "initial": language.cost(term),
        "greedy": greedy_cost(language, term),
        "quick": mcts(QUICK_BUDGET),
        "strong": strong,
    }


def is_hard(stats: dict) -> bool:
    return max(stats["greedy"], stats["quick"]) > stats["strong"]


def filter_hard(terms: Sequence[Expr], workers: int = 1) -> list[tuple[Expr, dict]]:
    """(term, stats) for the hard terms, in input order."""
    stats: Iterable[dict]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            stats = list(pool.map(hardness, terms, chunksize=16))
    else:
        stats = map(hardness, terms)
    return [(term, s) for term, s in zip(terms, stats) if is_hard(s)]
