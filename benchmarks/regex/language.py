from dataclasses import replace

from benchmarks.common import TreeLanguage, TreeRule

from .term import Alt, Empty, Eps, Expr, Seq, Star


def simplify(t: Expr) -> Expr | None:
    if t.op == "alt":
        a, b = t.args
        if a == b:
            return a
        if a.op == "empty":
            return b
        if b.op == "empty":
            return a
        if a.op == "eps" and b.op == "star":
            return b
        if b.op == "eps" and a.op == "star":
            return a
    if t.op == "seq":
        a, b = t.args
        if a.op == "empty" or b.op == "empty":
            return Empty()
        if a.op == "eps":
            return b
        if b.op == "eps":
            return a
        if a == b and a.op == "star":
            return a
    if t.op == "star":
        a = t.args[0]
        if a.op in ("empty", "eps"):
            return Eps()
        if a.op == "star":
            return a
        if a.op == "alt":
            if a.args[0].op == "eps":
                return Star(a.args[1])
            if a.args[1].op == "eps":
                return Star(a.args[0])
    return None


def fold_star(t: Expr) -> Expr | None:
    if t.op == "alt":
        for a, b in (t.args, tuple(reversed(t.args))):
            if b.op == "seq":
                left, right = b.args
                repeated = None
                if right.op == "star" and right.args[0] == left:
                    repeated = right
                if left.op == "star" and left.args[0] == right:
                    repeated = left
                if repeated is not None and (a.op == "eps" or a == repeated):
                    return repeated
    return None


def factor(t: Expr) -> Expr | None:
    if t.op == "alt" and all(a.op == "seq" for a in t.args):
        a, b = t.args
        if a.args[0] == b.args[0]:
            return Seq(a.args[0], Alt(a.args[1], b.args[1]))
        if a.args[1] == b.args[1]:
            return Seq(Alt(a.args[0], b.args[0]), a.args[1])
    return None


def distribute(t: Expr) -> Expr | None:
    if t.op == "seq":
        a, b = t.args
        if a.op == "alt":
            return Alt(Seq(a.args[0], b), Seq(a.args[1], b))
        if b.op == "alt":
            return Alt(Seq(a, b.args[0]), Seq(a, b.args[1]))
    return None


def associate(t: Expr, left: bool) -> Expr | None:
    if t.op in ("alt", "seq"):
        a, b = t.args
        make = Alt if t.op == "alt" else Seq
        if left and b.op == t.op:
            return make(make(a, b.args[0]), b.args[1])
        if not left and a.op == t.op:
            return make(a.args[0], make(a.args[1], b))
    return None


class RegexLanguage(TreeLanguage[Expr]):
    rules = (
        TreeRule("simplify", simplify),
        TreeRule("fold_star", fold_star),
        TreeRule("factor", factor),
        TreeRule("distribute", distribute),
        TreeRule("commute_alt", lambda t: Alt(t.args[1], t.args[0]) if t.op == "alt" else None),
        TreeRule("associate_left", lambda t: associate(t, True)),
        TreeRule("associate_right", lambda t: associate(t, False)),
    )

    def children(self, term: Expr) -> tuple[Expr, ...]:
        return term.args

    def rebuild(self, term: Expr, children: tuple[Expr, ...]) -> Expr:
        return replace(term, args=children)
