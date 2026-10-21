from dataclasses import dataclass
from math import isfinite, sqrt

from src.core import Engine, EngineConfig


@dataclass(frozen=True)
class MCTSConfig(EngineConfig):
    iterations: int = 1000
    rollout_depth: int = 12
    exploration_constant: float = sqrt(2)
    seed: int | None = 0
    max_tree_depth: int = 64
    max_states: int = 10_000
    time_limit: float | None = None

    def __post_init__(self) -> None:
        for name in ("iterations", "rollout_depth", "max_tree_depth", "max_states"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.iterations < 0 or self.rollout_depth < 0:
            raise ValueError("iterations and rollout_depth must be nonnegative")
        if self.max_tree_depth < 1 or self.max_states < 1:
            raise ValueError("max_tree_depth and max_states must be positive")
        if not isfinite(self.exploration_constant) or self.exploration_constant < 0:
            raise ValueError("exploration_constant must be finite and nonnegative")
        if self.time_limit is not None and (not isfinite(self.time_limit) or self.time_limit <= 0):
            raise ValueError("time_limit must be finite and positive")

    def create_engine(self) -> Engine:
        from .engine import MCTSEngine

        return MCTSEngine(self)
