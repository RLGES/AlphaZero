from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Generic, TypeVar

from src.language import Language

T = TypeVar("T")
A = TypeVar("A")


def checked_cost(language: Language[T, A], term: T) -> float:
    cost = float(language.cost(term))
    if not isfinite(cost) or cost < 0:
        raise ValueError("Language costs must be finite and nonnegative")
    return cost


@dataclass(frozen=True)
class TraceStep(Generic[T, A]):
    action: A
    before: T
    after: T
    before_cost: float
    after_cost: float


@dataclass(frozen=True)
class RewriteResult(Generic[T, A]):
    initial_term: T
    final_term: T
    initial_cost: float
    final_cost: float
    wall_time: float
    engine: str
    trace: tuple[TraceStep[T, A], ...] = ()
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def best_term(self) -> T:
        return self.final_term

    @property
    def improvement_percent(self) -> float:
        return (
            100 * (self.initial_cost - self.final_cost) / self.initial_cost
            if self.initial_cost
            else 0
        )


class Engine(ABC):
    @abstractmethod
    def run(self, language: Language[T, A], term: T) -> RewriteResult[T, A]:
        pass


class EngineConfig(ABC):
    @abstractmethod
    def create_engine(self) -> Engine:
        pass
