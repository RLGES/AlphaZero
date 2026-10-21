"""Egglog hooks are local to this backend; Language has no e-graph requirements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from egglog import BaseExpr, EGraph, Ruleset

T = TypeVar("T")


class EgglogAdapter(Protocol[T]):
    def encode(self, term: T) -> BaseExpr: ...
    def decode(self, term: BaseExpr) -> T: ...
    def ruleset(self) -> Ruleset: ...
    def cost_model(self, egraph: EGraph, expr: BaseExpr, child_costs: list[int]) -> int: ...
