"""Search knows no AST shape, matching policy, or language semantics."""

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")
A = TypeVar("A")


class Term(Protocol):
    """Marker for a language-owned IR. No universal tree interface is required."""


@dataclass(frozen=True)
class RewriteRule:
    name: str

    def __str__(self) -> str:
        return self.name


class Language(ABC, Generic[T, A]):
    """Deterministic rewrite semantics. Costs must be finite and nonnegative.

    Fingerprints must be hashable, stable, and distinguish semantically different
    search states. Terms need not be hashable. Applicable actions must be returned
    in stable order, and apply() must not mutate its input.
    """

    @abstractmethod
    def applicable(self, term: T) -> Sequence[A]:
        """Return available actions, excluding rewrites that leave the term unchanged."""

    @abstractmethod
    def apply(self, term: T, action: A) -> T:
        """Apply one language-defined action."""

    @abstractmethod
    def cost(self, term: T) -> float:
        """Lower is better."""

    @abstractmethod
    def fingerprint(self, term: T) -> Hashable:
        """Identity used for cycle detection and transpositions."""

    def format(self, term: T) -> str:
        return str(term)
