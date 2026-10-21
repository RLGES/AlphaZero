from dataclasses import replace

from benchmarks.common import TreeLanguage, TreeRule

from .term import And, Bool, Expr, Not, Or


def simplify(t: Expr) -> Expr | None:
    if t.op == "not":
        a = t.args[0]
        if a.op == "bool":
            return Bool(not a.value)
        if a.op == "not":
            return a.args[0]
    if t.op in ("and", "or"):
        a, b = t.args
        identity = Bool(t.op == "and")
        annihilator = Bool(t.op == "or")
        if a == b:
            return a
        if a == identity:
            return b
        if b == identity:
            return a
        if a == annihilator or b == annihilator:
            return annihilator
        if a == Not(b) or b == Not(a):
            return annihilator
        other = "or" if t.op == "and" else "and"
        if b.op == other and a in b.args:
            return a
        if a.op == other and b in a.args:
            return b
    return None


def demorgan(t: Expr) -> Expr | None:
    if t.op == "not" and t.args[0].op in ("and", "or"):
        a = t.args[0]
        make = Or if a.op == "and" else And
        return make(Not(a.args[0]), Not(a.args[1]))
    if t.op in ("and", "or") and all(a.op == "not" for a in t.args):
        make = Or if t.op == "and" else And
        return Not(make(t.args[0].args[0], t.args[1].args[0]))
    return None


def distribute(t: Expr) -> Expr | None:
    if t.op in ("and", "or"):
        a, b = t.args
        other = "or" if t.op == "and" else "and"
        make = And if t.op == "and" else Or
        outer = Or if t.op == "and" else And
        if b.op == other:
            return outer(make(a, b.args[0]), make(a, b.args[1]))
        if a.op == other:
            return outer(make(a.args[0], b), make(a.args[1], b))
    return None


def factor(t: Expr) -> Expr | None:
    if t.op in ("and", "or"):
        other = "or" if t.op == "and" else "and"
        a, b = t.args
        if a.op == b.op == other:
            for i in range(2):
                for j in range(2):
                    if a.args[i] == b.args[j]:
                        return Expr(other, (a.args[i], Expr(t.op, (a.args[1 - i], b.args[1 - j]))))
    return None


def associate(t: Expr, left: bool) -> Expr | None:
    if t.op in ("and", "or"):
        a, b = t.args
        if left and b.op == t.op:
            return Expr(t.op, (Expr(t.op, (a, b.args[0])), b.args[1]))
        if not left and a.op == t.op:
            return Expr(t.op, (a.args[0], Expr(t.op, (a.args[1], b))))
    return None


class BooleanLanguage(TreeLanguage[Expr]):
    rules = (
        TreeRule("simplify", simplify),
        TreeRule("demorgan", demorgan),
        TreeRule("distribute", distribute),
        TreeRule("factor", factor),
        TreeRule(
            "commute",
            lambda t: Expr(t.op, tuple(reversed(t.args))) if t.op in ("and", "or") else None,
        ),
        TreeRule("associate_left", lambda t: associate(t, True)),
        TreeRule("associate_right", lambda t: associate(t, False)),
    )

    def children(self, term: Expr) -> tuple[Expr, ...]:
        return term.args

    def rebuild(self, term: Expr, children: tuple[Expr, ...]) -> Expr:
        return replace(term, args=children)

    def egglog_adapter(self):
        from .egglog_adapter import BooleanAdapter

        return BooleanAdapter()
