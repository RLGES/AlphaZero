"""Public RewriteZero API; the source package deliberately lives directly in src/."""

from src.api import rewrite, rewrite_detailed
from src.core import Engine, EngineConfig, RewriteResult, TraceStep
from src.engines.alphazero import AlphaZeroConfig
from src.engines.egglog import EgglogConfig
from src.engines.mcts import MCTSConfig
from src.language import Language, RewriteRule, Term

__all__ = [
    "rewrite",
    "rewrite_detailed",
    "Engine",
    "EngineConfig",
    "RewriteResult",
    "TraceStep",
    "MCTSConfig",
    "AlphaZeroConfig",
    "EgglogConfig",
    "Language",
    "RewriteRule",
    "Term",
]
