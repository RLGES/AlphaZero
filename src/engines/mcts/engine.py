"""Small graph MCTS with UCT, cached transitions, and path-local cycle detection.

Repeated states share outgoing edges and transition/cost caches. UCT values are
stored on edges (not on their destination states), so two incoming actions do
not accidentally share visit counts. Only selected tree edges are backed up;
rollout transitions are cached but do not receive fictitious tree visits.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from math import log, sqrt
from random import Random
from time import perf_counter
from typing import Generic, TypeVar

from src.core import Engine, RewriteResult, TraceStep, checked_cost
from src.language import Language

from .config import MCTSConfig

T = TypeVar("T")
A = TypeVar("A")


@dataclass
class _State(Generic[T, A]):
    term: T
    key: Hashable
    cost: float
    actions: tuple[A, ...] | None = None
    untried: list[int] = field(default_factory=list)
    successors: dict[int, _State[T, A]] = field(default_factory=dict)
    edges: dict[int, _Edge[T, A]] = field(default_factory=dict)
    visits: int = 0


@dataclass
class _Edge(Generic[T, A]):
    target: _State[T, A]
    visits: int = 0
    reward_sum: float = 0.0


class MCTSEngine(Engine):
    def __init__(self, config: MCTSConfig):
        self.config = config

    def run(self, language: Language[T, A], term: T) -> RewriteResult[T, A]:
        start = perf_counter()
        cfg = self.config
        rng = Random(cfg.seed)
        initial_cost = checked_cost(language, term)
        root = _State(term, language.fingerprint(term), initial_cost)
        states: dict[Hashable, _State[T, A]] = {root.key: root}
        best = root
        best_trace: tuple[TraceStep[T, A], ...] = ()
        best_iteration = 0
        cycles = transpositions = transitions = rollout_steps = 0
        completed = 0
        stop_reason = "iterations"
        deadline = start + cfg.time_limit if cfg.time_limit is not None else float("inf")

        def actions(state: _State[T, A]) -> tuple[A, ...]:
            if state.actions is None:
                state.actions = tuple(language.applicable(state.term))
                state.untried = list(range(len(state.actions)))
            return state.actions

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

        def step(state: _State[T, A], index: int, target: _State[T, A]) -> TraceStep[T, A]:
            return TraceStep(
                actions(state)[index], state.term, target.term, state.cost, target.cost
            )

        def observe(state: _State[T, A], trace: list[TraceStep[T, A]], iteration: int) -> None:
            nonlocal best, best_trace, best_iteration
            if state.cost < best.cost:
                best = state
                best_trace = tuple(trace)
                best_iteration = iteration

        for iteration in range(1, cfg.iterations + 1):
            if perf_counter() >= deadline:
                stop_reason = "time_limit"
                break
            state = root
            seen = {root.key}
            path_states = [root]
            path_edges: list[_Edge[T, A]] = []
            trace: list[TraceStep[T, A]] = []
            reward_cost = root.cost
            cycle = False

            # Selection, followed by expansion of at most one tree edge.
            for _ in range(cfg.max_tree_depth):
                if perf_counter() >= deadline:
                    stop_reason = "time_limit"
                    break
                if not actions(state):
                    break
                expanding = bool(state.untried)
                if expanding:
                    index = state.untried.pop(rng.randrange(len(state.untried)))
                    target = successor(state, index)
                    if target is None:
                        break
                    edge = _Edge(target)
                    state.edges[index] = edge
                else:

                    def uct(item: tuple[int, _Edge[T, A]]) -> float:
                        e = item[1]
                        if not e.visits:
                            return float("inf")
                        return e.reward_sum / e.visits + cfg.exploration_constant * sqrt(
                            log(max(1, state.visits)) / e.visits
                        )

                    index, edge = max(state.edges.items(), key=uct)
                    target = edge.target
                path_edges.append(edge)
                trace.append(step(state, index, target))
                reward_cost = min(reward_cost, target.cost)
                observe(target, trace, iteration)
                if target.key in seen:
                    cycles += 1
                    cycle = True
                    break
                seen.add(target.key)
                state = target
                path_states.append(state)
                if expanding:
                    break

            # The reward uses the best encountered cost, including intermediates.
            if not cycle and stop_reason not in ("max_states", "time_limit"):
                for _ in range(cfg.rollout_depth):
                    if perf_counter() >= deadline:
                        stop_reason = "time_limit"
                        break
                    available = actions(state)
                    if not available:
                        break
                    index = rng.randrange(len(available))
                    target = successor(state, index)
                    if target is None:
                        break
                    rollout_steps += 1
                    trace.append(step(state, index, target))
                    reward_cost = min(reward_cost, target.cost)
                    observe(target, trace, iteration)
                    if target.key in seen:
                        cycles += 1
                        break
                    seen.add(target.key)
                    state = target

            # Normalize improvement to [0, 1], independent of language cost scale.
            reward = (initial_cost - reward_cost) / initial_cost if initial_cost else 0.0
            for visited in path_states:
                visited.visits += 1
            for edge in path_edges:
                edge.visits += 1
                edge.reward_sum += reward
            completed = iteration
            if stop_reason in ("max_states", "time_limit"):
                break

        return RewriteResult(
            term,
            best.term,
            initial_cost,
            best.cost,
            perf_counter() - start,
            "MCTS",
            best_trace,
            {
                "iterations": completed,
                "states_explored": len(states),
                "best_iteration": best_iteration,
                "transpositions": transpositions,
                "cycles_detected": cycles,
                "transitions": transitions,
                "rollout_steps": rollout_steps,
                "stop_reason": stop_reason,
            },
        )
