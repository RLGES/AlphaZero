from random import Random
from time import perf_counter
from typing import TypeVar

from src.core import Engine, RewriteResult
from src.engines.egglog.engine import UnsupportedLanguageError
from src.language import Language

from .config import AlphaZeroConfig
from .features import supports_alphazero
from .search import Evaluator, PUCTSearch, UniformEvaluator

T = TypeVar("T")
A = TypeVar("A")


class AlphaZeroEngine(Engine):
    def __init__(self, config: AlphaZeroConfig):
        self.config = config

    def _evaluator(self, language: Language[T, A]) -> Evaluator:
        cfg = self.config
        if cfg.network is None and cfg.checkpoint is None:
            return UniformEvaluator()
        try:
            from .inference import NetworkEvaluator
        except ImportError as exc:
            raise RuntimeError(
                "AlphaZero networks need PyTorch. Install it with: pip install -e '.[alphazero]'"
            ) from exc
        if cfg.network is not None:
            return NetworkEvaluator(cfg.network, cfg.device)
        from .checkpoint import load_network

        return NetworkEvaluator(load_network(cfg.checkpoint, language, cfg.device), cfg.device)

    def run(self, language: Language[T, A], term: T) -> RewriteResult[T, A]:
        start = perf_counter()
        if not supports_alphazero(language):
            raise UnsupportedLanguageError(
                f"{type(language).__name__} provides no features()/action_vocabulary(); "
                "AlphaZero cannot search it."
            )
        search = PUCTSearch(
            language, self._evaluator(language), self.config, Random(self.config.seed)
        )
        result = search.run(term)
        return RewriteResult(
            term,
            result.best_term,
            result.initial_cost,
            result.best_cost,
            perf_counter() - start,
            "AlphaZero",
            result.trace,
            result.statistics,
        )
