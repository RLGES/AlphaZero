"""Dimension-checked exact matrices; multiplication is deliberately noncommutative."""

from dataclasses import dataclass

MatrixValue = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Expr:
    op: str
    rows: int
    cols: int
    args: tuple["Expr", ...] = ()
    name: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.rows) is not int
            or type(self.cols) is not int
            or min(self.rows, self.cols) < 1
        ):
            raise ValueError("Matrix dimensions must be positive integers")
        if self.op == "matrix":
            valid = isinstance(self.name, str) and not self.args
        elif self.op in ("zero", "identity"):
            valid = not self.args and self.name is None
            if self.op == "identity":
                valid = valid and self.rows == self.cols
        elif self.op == "transpose":
            valid = len(self.args) == 1 and self.name is None
            if valid:
                a = self.args[0]
                valid = (self.rows, self.cols) == (a.cols, a.rows)
        elif self.op in ("add", "matmul"):
            valid = len(self.args) == 2 and self.name is None
            if valid:
                a, b = self.args
                if self.op == "add":
                    valid = (a.rows, a.cols) == (b.rows, b.cols) == (self.rows, self.cols)
                else:
                    valid = a.cols == b.rows and (self.rows, self.cols) == (a.rows, b.cols)
        else:
            valid = False
        if not valid:
            raise ValueError("Invalid matrix term or incompatible dimensions")

    def __str__(self) -> str:
        if self.op == "matrix":
            return f"{self.name}[{self.rows}x{self.cols}]"
        if self.op in ("identity", "zero"):
            return f"{self.op}[{self.rows}x{self.cols}]"
        if self.op == "transpose":
            return f"T({self.args[0]})"
        return f"({self.args[0]} {'@' if self.op == 'matmul' else '+'} {self.args[1]})"


def Matrix(name: str, rows: int, cols: int) -> Expr:
    return Expr("matrix", rows, cols, name=name)


def Identity(n: int) -> Expr:
    return Expr("identity", n, n)


def Zero(rows: int, cols: int) -> Expr:
    return Expr("zero", rows, cols)


def Add(a: Expr, b: Expr) -> Expr:
    return Expr("add", a.rows, a.cols, (a, b))


def MatMul(a: Expr, b: Expr) -> Expr:
    return Expr("matmul", a.rows, b.cols, (a, b))


def Transpose(a: Expr) -> Expr:
    return Expr("transpose", a.cols, a.rows, (a,))


def evaluate(t: Expr, env: dict[str, MatrixValue]) -> MatrixValue:
    if t.op == "matrix":
        value = env[t.name]
        if len(value) != t.rows or any(len(row) != t.cols for row in value):
            raise ValueError("Environment matrix has the wrong shape")
        return value
    if t.op in ("identity", "zero"):
        return tuple(
            tuple(int(t.op == "identity" and i == j) for j in range(t.cols)) for i in range(t.rows)
        )
    if t.op == "transpose":
        return tuple(zip(*evaluate(t.args[0], env)))
    a, b = (evaluate(child, env) for child in t.args)
    if t.op == "add":
        return tuple(tuple(x + y for x, y in zip(ar, br)) for ar, br in zip(a, b))
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(t.args[0].cols)) for j in range(t.cols))
        for i in range(t.rows)
    )
