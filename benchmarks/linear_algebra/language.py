from dataclasses import replace

from benchmarks.common import TreeLanguage, TreeRule

from .term import Add, Expr, MatMul, Transpose, Zero


def simplify(t: Expr) -> Expr | None:
    if t.op == "transpose":
        a = t.args[0]
        if a.op == "transpose":
            return a.args[0]
        if a.op == "zero":
            return Zero(t.rows, t.cols)
        if a.op == "identity":
            return a
    if t.op in ("add", "matmul"):
        a, b = t.args
        if t.op == "add":
            if a.op == "zero":
                return b
            if b.op == "zero":
                return a
        else:
            if a.op == "zero" or b.op == "zero":
                return Zero(t.rows, t.cols)
            if a.op == "identity":
                return b
            if b.op == "identity":
                return a
    return None


def associate(t: Expr, left: bool) -> Expr | None:
    if t.op in ("add", "matmul"):
        a, b = t.args
        make = Add if t.op == "add" else MatMul
        if left and b.op == t.op:
            return make(make(a, b.args[0]), b.args[1])
        if not left and a.op == t.op:
            return make(a.args[0], make(a.args[1], b))
    return None


def transpose(t: Expr) -> Expr | None:
    if t.op == "transpose":
        a = t.args[0]
        if a.op == "add":
            return Add(Transpose(a.args[0]), Transpose(a.args[1]))
        if a.op == "matmul":
            return MatMul(Transpose(a.args[1]), Transpose(a.args[0]))
    return None


def distribute(t: Expr) -> Expr | None:
    if t.op == "matmul":
        a, b = t.args
        if a.op == "add":
            return Add(MatMul(a.args[0], b), MatMul(a.args[1], b))
        if b.op == "add":
            return Add(MatMul(a, b.args[0]), MatMul(a, b.args[1]))
    return None


def factor(t: Expr) -> Expr | None:
    if t.op == "add" and all(a.op == "matmul" for a in t.args):
        a, b = t.args
        if a.args[0] == b.args[0]:
            return MatMul(a.args[0], Add(a.args[1], b.args[1]))
        if a.args[1] == b.args[1]:
            return MatMul(Add(a.args[0], b.args[0]), a.args[1])
    return None


class LinearAlgebraLanguage(TreeLanguage[Expr]):
    rules = (
        TreeRule("simplify", simplify),
        TreeRule("transpose", transpose),
        TreeRule("associate_left", lambda t: associate(t, True)),
        TreeRule("associate_right", lambda t: associate(t, False)),
        TreeRule("distribute", distribute),
        TreeRule("factor", factor),
        TreeRule("commute_add", lambda t: Add(t.args[1], t.args[0]) if t.op == "add" else None),
    )

    def children(self, term: Expr) -> tuple[Expr, ...]:
        return term.args

    def rebuild(self, term: Expr, children: tuple[Expr, ...]) -> Expr:
        return replace(term, args=children)

    def node_label(self, term: Expr) -> str:
        # Dimensions drive the cost model, so they are part of the label.
        return f"{term.op}:{term.rows}x{term.cols}"

    def cost(self, term: Expr) -> float:
        child_cost = sum(self.cost(child) for child in term.args)
        if term.op == "matmul":
            work = term.rows * term.args[0].cols * term.cols
        elif term.op in ("add", "transpose"):
            work = term.rows * term.cols
        else:
            work = 1
        return float(work + child_cost)
