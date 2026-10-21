from dataclasses import replace

from benchmarks.common import TreeLanguage, TreeRule

from .term import Add, Const, Expr, Mul


def simplify(t: Expr) -> Expr | None:
    if not t.args:
        return None
    a, b = t.args
    if a.op == b.op == "const":
        return Const(a.value + b.value if t.op == "add" else a.value * b.value)
    if t.op == "add":
        if a == Const(0):
            return b
        if b == Const(0):
            return a
    if t.op == "mul":
        if a == Const(0) or b == Const(0):
            return Const(0)
        if a == Const(1):
            return b
        if b == Const(1):
            return a
    return None


def factor(t: Expr) -> Expr | None:
    if t.op == "add" and all(a.op == "mul" for a in t.args):
        a, b = t.args
        for i in range(2):
            for j in range(2):
                if a.args[i] == b.args[j]:
                    return Mul(a.args[i], Add(a.args[1 - i], b.args[1 - j]))
    return None


def distribute(t: Expr) -> Expr | None:
    if t.op == "mul":
        a, b = t.args
        if b.op == "add":
            return Add(Mul(a, b.args[0]), Mul(a, b.args[1]))
        if a.op == "add":
            return Add(Mul(a.args[0], b), Mul(a.args[1], b))
    return None


def associate(t: Expr, left: bool) -> Expr | None:
    if t.op not in ("add", "mul"):
        return None
    a, b = t.args
    make = Add if t.op == "add" else Mul
    if left and b.op == t.op:
        return make(make(a, b.args[0]), b.args[1])
    if not left and a.op == t.op:
        return make(a.args[0], make(a.args[1], b))
    return None


class ArithmeticLanguage(TreeLanguage[Expr]):
    rules = (
        TreeRule("simplify", simplify),
        TreeRule("factor", factor),
        TreeRule("distribute", distribute),
        TreeRule("commute", lambda t: Expr(t.op, tuple(reversed(t.args))) if t.args else None),
        TreeRule("associate_left", lambda t: associate(t, True)),
        TreeRule("associate_right", lambda t: associate(t, False)),
    )

    def children(self, term: Expr) -> tuple[Expr, ...]:
        return term.args

    def rebuild(self, term: Expr, children: tuple[Expr, ...]) -> Expr:
        return replace(term, args=children)

    def egglog_adapter(self):
        from .egglog_adapter import ArithmeticAdapter

        return ArithmeticAdapter()
