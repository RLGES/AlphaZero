"""Neural search sees terms only through language-provided graph features.

A language opts into AlphaZero by implementing ``features(term)`` and
``action_vocabulary()``. The vocabulary is a fixed, ordered superset of every
``applicable(term)`` result, so the policy can score actions by position.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphFeatures:
    labels: tuple[str, ...]
    edges: tuple[tuple[int, int, int], ...]  # (parent, child, child position)
    root: int = 0


def supports_alphazero(language: object) -> bool:
    return callable(getattr(language, "features", None)) and callable(
        getattr(language, "action_vocabulary", None)
    )
