"""Self-play episodes (PPT slide 16, PlayGame) and the replay buffer.

Each move runs a PUCT search from the current state; the root visit counts
become the policy target. A state's value target is the fraction of its cost
removed by the best cost reached afterwards, counting both the states actually
played and the best states each later search found (all reachable from it).
This module needs no PyTorch, so self-play workers can stay lightweight.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from random import Random
from typing import Any, TypeVar

from src.language import Language

from .config import AlphaZeroConfig
from .features import GraphFeatures
from .search import Evaluator, PUCTSearch

T = TypeVar("T")
A = TypeVar("A")


@dataclass(frozen=True)
class Sample:
    features: GraphFeatures
    mask: tuple[bool, ...]
    policy: tuple[float, ...]
    value: float


@dataclass(frozen=True)
class SelfPlayConfig:
    simulations_per_move: int = 64
    max_moves: int = 24
    temperature_moves: int = 8
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.5

    def search_config(self) -> AlphaZeroConfig:
        return AlphaZeroConfig(
            iterations=self.simulations_per_move,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
        )


def play_episode(
    language: Language[T, A],
    term: T,
    evaluator: Evaluator,
    config: SelfPlayConfig,
    rng: Random,
) -> list[Sample]:
    vocabulary = tuple(language.action_vocabulary())
    search_config = config.search_config()
    records: list[tuple[GraphFeatures, tuple[bool, ...], tuple[float, ...]]] = []
    costs: list[float] = []  # cost of each recorded state
    reachable: list[float] = []  # best cost known reachable from each recorded state
    state = term
    seen = {language.fingerprint(state)}

    for move in range(config.max_moves):
        applicable = set(language.applicable(state))
        if not applicable:
            break
        result = PUCTSearch(language, evaluator, search_config, rng).run(state)
        total = sum(result.visits)
        if not total:
            break
        records.append(
            (
                language.features(state),
                tuple(action in applicable for action in vocabulary),
                tuple(visits / total for visits in result.visits),
            )
        )
        costs.append(result.initial_cost)
        reachable.append(result.best_cost)
        indices = range(len(vocabulary))
        if move < config.temperature_moves:
            index = rng.choices(indices, weights=result.visits)[0]
        else:
            index = max(indices, key=result.visits.__getitem__)
        state = language.apply(state, vocabulary[index])
        key = language.fingerprint(state)
        if key in seen:
            break
        seen.add(key)

    samples: list[Sample] = []
    best_future = float("inf")
    for (features, mask, policy), cost, best in zip(
        reversed(records), reversed(costs), reversed(reachable)
    ):
        best_future = min(best_future, best, cost)
        value = (cost - best_future) / cost if cost else 0.0
        samples.append(Sample(features, mask, policy, value))
    samples.reverse()
    return samples


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._items: deque[Any] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def extend(self, samples: Iterable[Any]) -> None:
        self._items.extend(samples)

    def sample(self, batch_size: int, rng: Random) -> list[Any]:
        if not self._items:
            return []
        if batch_size <= len(self._items):
            return rng.sample(list(self._items), batch_size)
        return rng.choices(list(self._items), k=batch_size)

    def items(self) -> list[Any]:
        return list(self._items)
