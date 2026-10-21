from dataclasses import dataclass


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value: bool | str | None = None

    def __post_init__(self) -> None:
        arity = {"not": 1, "and": 2, "or": 2}
        valid = (
            (self.op == "bool" and type(self.value) is bool and not self.args)
            or (self.op == "var" and isinstance(self.value, str) and not self.args)
            or (self.op in arity and len(self.args) == arity[self.op] and self.value is None)
        )
        if not valid:
            raise ValueError("Invalid Boolean term")

    def __str__(self) -> str:
        if not self.args:
            return str(self.value).lower() if self.op == "bool" else str(self.value)
        if self.op == "not":
            return f"!{self.args[0]}"
        return f"({self.args[0]} {'&' if self.op == 'and' else '|'} {self.args[1]})"


def Bool(value: bool) -> Expr:
    return Expr("bool", value=value)


def Var(name: str) -> Expr:
    return Expr("var", value=name)


def Not(a: Expr) -> Expr:
    return Expr("not", (a,))


def And(a: Expr, b: Expr) -> Expr:
    return Expr("and", (a, b))


def Or(a: Expr, b: Expr) -> Expr:
    return Expr("or", (a, b))


def evaluate(t: Expr, env: dict[str, bool]) -> bool:
    if t.op == "bool":
        return bool(t.value)
    if t.op == "var":
        return env[str(t.value)]
    if t.op == "not":
        return not evaluate(t.args[0], env)
    a, b = (evaluate(child, env) for child in t.args)
    return a and b if t.op == "and" else a or b
