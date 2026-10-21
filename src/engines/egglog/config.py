from dataclasses import dataclass

from src.core import Engine, EngineConfig


@dataclass(frozen=True)
class EgglogConfig(EngineConfig):
    iterations: int = 6
    match_limit: int = 1000

    def __post_init__(self) -> None:
        if type(self.iterations) is not int or self.iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        if type(self.match_limit) is not int or self.match_limit < 1:
            raise ValueError("match_limit must be a positive integer")

    def create_engine(self) -> Engine:
        from .engine import EgglogEngine

        return EgglogEngine(self)
