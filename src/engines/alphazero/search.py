"""PUCT search over a language's state graph, guided by an evaluator.

The state/edge layout follows the MCTS engine: transpositions share states and
cached transitions, statistics live on edges, and cycles are detected against
the current path. Random rollouts are replaced by one evaluator call per new
leaf, which supplies priors over the action vocabulary and a value in [0, 1]:
the fraction of the leaf's cost that can still be removed.

A path's reward uses the same scale as MCTS, (initial - best) / initial, where
the leaf contributes the estimate ``cost * (1 - value)``. Only real costs ever
update the reported best term.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from math import sqrt
from random import Random
from time import perf_counter
from typing import Any, Generic, Protocol, TypeVar

from src.core import TraceStep, checked_cost
from src.language import Language

from .config import AlphaZeroConfig

T = TypeVar("T")
A = TypeVar("A")


class Evaluator(Protocol):
    def evaluate(
        self, language: Language[T, A], terms: Sequence[T]
    ) -> list[tuple[Sequence[float], float]]:
        """Return (prior over action_vocabulary(), value in [0, 1]) per term."""


class UniformEvaluator:
    """Untrained stand-in: uniform priors and no value information."""

    def evaluate(
        self, language: Language[T, A], terms: Sequence[T]
    ) -> list[tuple[Sequence[float], float]]:
        width = len(language.action_vocabulary())
        return [([1.0] * width, 0.0) for _ in terms]


@dataclass
class _State(Generic[T, A]):
    term: T
    key: Hashable
    cost: float
    actions: tuple[A, ...] | None = None
    action_ids: tuple[int, ...] = ()
    priors: list[float] | None = None  # None until the evaluator has seen this state
    successors: dict[int, _State[T, A]] = field(default_factory=dict)
    edges: dict[int, _Edge] = field(default_factory=dict)
    visits: int = 0


@dataclass
class _Edge:
    visits: int = 0
    reward_sum: float = 0.0


@dataclass(frozen=True)
class SearchResult(Generic[T, A]):
    visits: tuple[int, ...]  # root edge visits, indexed like action_vocabulary()
    initial_cost: float
    best_term: T
    best_cost: float
    trace: tuple[TraceStep[T, A], ...]
    statistics: dict[str, Any]


class PUCTSearch(Generic[T, A]):
    def __init__(
        self,
        language: Language[T, A],
        evaluator: Evaluator,
        config: AlphaZeroConfig,
        rng: Random,
    ):
        self.language = language
        self.evaluator = evaluator
        self.config = config
        self.rng = rng
        self.vocabulary = tuple(language.action_vocabulary())
        self.index = {action: i for i, action in enumerate(self.vocabulary)}

    def run(self, term: T) -> SearchResult[T, A]:
        start = perf_counter()
        cfg = self.config
        language = self.language
        initial_cost = checked_cost(language, term)
        root = _State(term, language.fingerprint(term), initial_cost)
        states: dict[Hashable, _State[T, A]] = {root.key: root}
        best = root
        best_trace: tuple[TraceStep[T, A], ...] = ()
        best_iteration = 0
        cycles = transpositions = transitions = evaluations = 0
        completed = 0
        stop_reason = "iterations"
        deadline = start + cfg.time_limit if cfg.time_limit is not None else float("inf")

        def actions(state: _State[T, A]) -> tuple[A, ...]:
            if state.actions is None:
                state.actions = tuple(language.applicable(state.term))
                try:
                    state.action_ids = tuple(self.index[a] for a in state.actions)
                except KeyError as exc:
                    raise ValueError(
                        "applicable() returned an action outside action_vocabulary()"
                    ) from exc
            return state.actions

        def evaluate(state: _State[T, A]) -> float:
            nonlocal evaluations
            if not actions(state):
                state.priors = []
                return 0.0
            ((prior, value),) = self.evaluator.evaluate(language, [state.term])
            evaluations += 1
            raw = [max(0.0, float(prior[i])) for i in state.action_ids]
            total = sum(raw)
            count = len(raw)
            state.priors = [p / total for p in raw] if total > 0 else [1.0 / count] * count
            return min(1.0, max(0.0, float(value)))

        def successor(state: _State[T, A], index: int) -> _State[T, A] | None:
            nonlocal transpositions, transitions, stop_reason
            if index in state.successors:
                return state.successors[index]
            if len(states) >= cfg.max_states:
                stop_reason = "max_states"
                return None
            new_term = language.apply(state.term, actions(state)[index])
            transitions += 1
            key = language.fingerprint(new_term)
            if key in states:
                target = states[key]
                transpositions += 1
            else:
                target = _State(new_term, key, checked_cost(language, new_term))
                states[key] = target
            state.successors[index] = target
            return target

        def select(state: _State[T, A]) -> int:
            scale = cfg.c_puct * sqrt(max(1, state.visits))

            def score(index: int) -> float:
                edge = state.edges.get(index)
                visits = edge.visits if edge else 0
                q = edge.reward_sum / visits if visits else 0.0
                return q + scale * state.priors[index] / (1 + visits)

            return max(range(len(state.priors)), key=score)

        if cfg.iterations and actions(root):
            evaluate(root)
            if cfg.dirichlet_epsilon > 0 and len(root.priors) > 1:
                noise = [self.rng.gammavariate(cfg.dirichlet_alpha, 1.0) for _ in root.priors]
                total = sum(noise) or 1.0
                eps = cfg.dirichlet_epsilon
                root.priors = [(1 - eps) * p + eps * n / total for p, n in zip(root.priors, noise)]

            for iteration in range(1, cfg.iterations + 1):
                if perf_counter() >= deadline:
                    stop_reason = "time_limit"
                    break
                state = root
                seen = {root.key}
                path_states = [root]
                path_edges: list[_Edge] = []
                trace: list[TraceStep[T, A]] = []
                path_cost = root.cost
                cycle = False

                for _ in range(cfg.max_tree_depth):
                    if perf_counter() >= deadline:
                        stop_reason = "time_limit"
                        break
                    if state.priors is None or not state.actions:
                        break
                    index = select(state)
                    target = successor(state, index)
                    if target is None:
                        break
                    path_edges.append(state.edges.setdefault(index, _Edge()))
                    trace.append(
                        TraceStep(
                            state.actions[index], state.term, target.term, state.cost, target.cost
                        )
                    )
                    path_cost = min(path_cost, target.cost)
                    if target.cost < best.cost:
                        best, best_trace, best_iteration = target, tuple(trace), iteration
                    if target.key in seen:
                        cycles += 1
                        cycle = True
                        break
                    seen.add(target.key)
                    state = target
                    path_states.append(state)

                if not cycle and state.priors is None:
                    value = evaluate(state)
                    path_cost = min(path_cost, state.cost * (1 - value))

                reward = (initial_cost - path_cost) / initial_cost if initial_cost else 0.0
                for visited in path_states:
                    visited.visits += 1
                for edge in path_edges:
                    edge.visits += 1
                    edge.reward_sum += reward
                completed = iteration
                if stop_reason in ("max_states", "time_limit"):
                    break

        visits = [0] * len(self.vocabulary)
        for index, edge in root.edges.items():
            visits[root.action_ids[index]] = edge.visits
        return SearchResult(
            tuple(visits),
            initial_cost,
            best.term,
            best.cost,
            best_trace,
            {
                "iterations": completed,
                "states_explored": len(states),
                "best_iteration": best_iteration,
                "transpositions": transpositions,
                "cycles_detected": cycles,
                "transitions": transitions,
                "network_evaluations": evaluations,
                "stop_reason": stop_reason,
            },
        )
