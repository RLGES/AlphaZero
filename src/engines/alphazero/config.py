from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from src.core import Engine, EngineConfig


@dataclass(frozen=True)
class AlphaZeroConfig(EngineConfig):
    """PUCT search guided by a policy/value network.

    ``network`` (an in-memory PolicyValueNet, used by self-play) wins over
    ``checkpoint``. With neither, priors are uniform and values are zero, which
    needs no PyTorch.
    """

    iterations: int = 200
    c_puct: float = 1.5
    checkpoint: str | None = None
    seed: int | None = 0
    max_tree_depth: int = 64
    max_states: int = 10_000
    time_limit: float | None = None
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0
    device: str = "cpu"
    network: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("iterations", "max_tree_depth", "max_states"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.iterations < 0:
            raise ValueError("iterations must be nonnegative")
        if self.max_tree_depth < 1 or self.max_states < 1:
            raise ValueError("max_tree_depth and max_states must be positive")
        if not isfinite(self.c_puct) or self.c_puct < 0:
            raise ValueError("c_puct must be finite and nonnegative")
        if not isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be finite and positive")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if self.time_limit is not None and (not isfinite(self.time_limit) or self.time_limit <= 0):
            raise ValueError("time_limit must be finite and positive")

    def create_engine(self) -> Engine:
        from .engine import AlphaZeroEngine

        return AlphaZeroEngine(self)
