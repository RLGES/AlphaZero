from time import perf_counter
from typing import TypeVar

from src.core import Engine, RewriteResult, checked_cost
from src.language import Language

from .config import EgglogConfig

T = TypeVar("T")
A = TypeVar("A")


class UnsupportedLanguageError(ValueError):
    pass


class EgglogUnavailableError(RuntimeError):
    pass


class EgglogEngine(Engine):
    def __init__(self, config: EgglogConfig):
        self.config = config

    def run(self, language: Language[T, A], term: T) -> RewriteResult[T, A]:
        start = perf_counter()
        factory = getattr(language, "egglog_adapter", None)
        if not callable(factory):
            raise UnsupportedLanguageError(
                f"{type(language).__name__} has no egglog adapter; this suite supports MCTS only."
            )
        try:
            from egglog import EGraph, back_off, run
        except ImportError as exc:
            raise EgglogUnavailableError(
                "egglog is optional. Install it with: pip install -e '.[egglog]'"
            ) from exc

        initial_cost = checked_cost(language, term)
        adapter = factory()
        egraph = EGraph()
        root = adapter.encode(term)
        egraph.register(root)
        rules = adapter.ruleset()
        setup_time = perf_counter() - start
        report = egraph.run(
            run(rules, scheduler=back_off(match_limit=self.config.match_limit, ban_length=2))
            * self.config.iterations
        )
        extracted, extraction_cost = egraph.extract(
            root, include_cost=True, cost_model=adapter.cost_model
        )
        candidate = adapter.decode(extracted)
        candidate_cost = checked_cost(language, candidate)
        # Report the same language cost as MCTS and retain the initial term on ties.
        final = candidate if candidate_cost < initial_cost else term
        return RewriteResult(
            term,
            final,
            initial_cost,
            min(initial_cost, candidate_cost),
            perf_counter() - start,
            "egglog",
            statistics={
                "iterations": len(report.iterations),
                "rule_matches": sum(report.num_matches_per_rule.values()),
                "extraction_cost": extraction_cost,
                "setup_time": setup_time,
                "trace_available": False,
            },
        )
