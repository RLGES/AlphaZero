"""Tree conveniences for the benchmark languages, never used by generic search."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from src.engines.alphazero.features import GraphFeatures
from src.engines.mcts import MCTSConfig
from src.language import Language, RewriteRule

T = TypeVar("T")


@dataclass(frozen=True)
class TreeRule(RewriteRule, Generic[T]):
    transform: Callable[[T], T | None]


class TreeLanguage(Language[T, TreeRule[T]]):
    """A rule rewrites original matches top-down, left-to-right.

    A replaced root blocks its descendants during this action. New subterms are
    not revisited. Disjoint sibling matches are all rewritten in the same batch.
    The finite node limit prevents algebraic expansion from exhausting memory;
    it restricts available actions, not the equivalence relation.
    """

    rules: tuple[TreeRule[T], ...] = ()

    def __init__(self, max_nodes: int = 256):
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.max_nodes = max_nodes

    def children(self, term: T) -> tuple[T, ...]:
        raise NotImplementedError

    def rebuild(self, term: T, children: tuple[T, ...]) -> T:
        raise NotImplementedError

    def size(self, term: T) -> int:
        count = 0
        pending = [term]
        while pending:
            node = pending.pop()
            count += 1
            pending.extend(self.children(node))
        return count

    def _batch(self, term: T, rule: TreeRule[T]) -> T:
        replacement = rule.transform(term)
        if replacement is not None and replacement != term:
            return replacement
        old = self.children(term)
        new = tuple(self._batch(child, rule) for child in old)
        return self.rebuild(term, new) if new != old else term

    def apply(self, term: T, action: TreeRule[T]) -> T:
        if action not in self.rules:
            raise ValueError(f"Unknown rule: {action.name}")
        candidate = self._batch(term, action)
        return candidate if self.size(candidate) <= self.max_nodes else term

    def applicable(self, term: T) -> Sequence[TreeRule[T]]:
        return tuple(rule for rule in self.rules if self.apply(term, rule) != term)

    def cost(self, term: T) -> float:
        return float(self.size(term))

    def fingerprint(self, term: T) -> Hashable:
        # These benchmark IRs are frozen dataclasses; arbitrary user IRs need not be.
        return term

    def node_label(self, term: T) -> str:
        """Operator plus small literal payloads; variable names carry no semantics."""
        label = str(getattr(term, "op", type(term).__name__))
        value = getattr(term, "value", None)
        if isinstance(value, bool):
            return f"{label}:{value}"
        if isinstance(value, int):
            return f"{label}:{value}" if abs(value) <= 16 else f"{label}:big"
        return label

    def features(self, term: T) -> GraphFeatures:
        labels: list[str] = []
        edges: list[tuple[int, int, int]] = []
        pending: list[tuple[T, int, int]] = [(term, -1, 0)]
        while pending:
            node, parent, position = pending.pop()
            index = len(labels)
            labels.append(self.node_label(node))
            if parent >= 0:
                edges.append((parent, index, position))
            children = self.children(node)
            pending.extend((child, index, i) for i, child in reversed(tuple(enumerate(children))))
        return GraphFeatures(tuple(labels), tuple(edges))

    def action_vocabulary(self) -> tuple[TreeRule[T], ...]:
        return self.rules


@dataclass(frozen=True)
class Benchmark(Generic[T]):
    name: str
    term: T
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    config: MCTSConfig = MCTSConfig()


@dataclass(frozen=True)
class Suite(Generic[T]):
    name: str
    language: TreeLanguage[T]
    benchmarks: tuple[Benchmark[T], ...]
