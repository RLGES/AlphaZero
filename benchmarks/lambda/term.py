from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class Var:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Abs:
    parameter: str
    body: Term

    def __str__(self) -> str:
        return f"(lambda {self.parameter}. {self.body})"


@dataclass(frozen=True)
class App:
    function: Term
    argument: Term

    def __str__(self) -> str:
        return f"({self.function} {self.argument})"


Term: TypeAlias = Var | Abs | App


def free_vars(t: Term) -> frozenset[str]:
    if isinstance(t, Var):
        return frozenset((t.name,))
    if isinstance(t, Abs):
        return free_vars(t.body) - {t.parameter}
    return free_vars(t.function) | free_vars(t.argument)


def all_names(t: Term) -> frozenset[str]:
    if isinstance(t, Var):
        return frozenset((t.name,))
    if isinstance(t, Abs):
        return all_names(t.body) | {t.parameter}
    return all_names(t.function) | all_names(t.argument)


def substitute(t: Term, name: str, value: Term) -> Term:
    if isinstance(t, Var):
        return value if t.name == name else t
    if isinstance(t, App):
        return App(substitute(t.function, name, value), substitute(t.argument, name, value))
    if t.parameter == name or name not in free_vars(t.body):
        return t
    body = t.body
    parameter = t.parameter
    if parameter in free_vars(value):
        forbidden = all_names(body) | all_names(value) | {name}
        suffix = 0
        while f"{parameter}_{suffix}" in forbidden:
            suffix += 1
        fresh = f"{parameter}_{suffix}"
        body = substitute(body, parameter, Var(fresh))
        parameter = fresh
    return Abs(parameter, substitute(body, name, value))
