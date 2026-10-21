from typing import TypeVar

from src.core import EngineConfig, RewriteResult
from src.language import Language

T = TypeVar("T")
A = TypeVar("A")


def rewrite_detailed(
    language: Language[T, A], term: T, engine_config: EngineConfig | None = None
) -> RewriteResult[T, A]:
    if engine_config is None:
        from src.engines.mcts import MCTSConfig

        engine_config = MCTSConfig()
    return engine_config.create_engine().run(language, term)


def rewrite(language: Language[T, A], term: T, engine_config: EngineConfig | None = None) -> T:
    return rewrite_detailed(language, term, engine_config).final_term
