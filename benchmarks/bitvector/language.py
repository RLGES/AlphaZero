from dataclasses import replace

from benchmarks.common import TreeLanguage, TreeRule

from .term import Const, Expr, Not, evaluate


def simplify(t: Expr) -> Expr | None:
    if not t.args:
        return None
    if all(a.op == "const" for a in t.args):
        return Const(evaluate(t, {}), t.width)
    if t.op == "not":
        if t.args[0].op == "not":
            return t.args[0].args[0]
        return None
    a, b = t.args
    zero, ones = Const(0, t.width), Const(t.mask, t.width)
    if t.op in ("xor", "add"):
        if a == zero:
            return b
        if b == zero:
            return a
    if t.op == "xor":
        if a == b:
            return zero
        if a == ones:
            return Not(b)
        if b == ones:
            return Not(a)
    if t.op in ("and", "or"):
        identity, annihilator = (ones, zero) if t.op == "and" else (zero, ones)
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


def associate(t: Expr, left: bool) -> Expr | None:
    if t.op in ("and", "or", "xor", "add"):
        a, b = t.args
        if left and b.op == t.op:
            return Expr(t.op, t.width, (Expr(t.op, t.width, (a, b.args[0])), b.args[1]))
        if not left and a.op == t.op:
            return Expr(t.op, t.width, (a.args[0], Expr(t.op, t.width, (a.args[1], b))))
    return None


def demorgan(t: Expr) -> Expr | None:
    if t.op == "not" and t.args[0].op in ("and", "or"):
        a = t.args[0]
        other = "or" if a.op == "and" else "and"
        return Expr(other, t.width, tuple(Not(child) for child in a.args))
    if t.op in ("and", "or") and all(a.op == "not" for a in t.args):
        other = "or" if t.op == "and" else "and"
        return Not(Expr(other, t.width, tuple(a.args[0] for a in t.args)))
    return None


class BitvectorLanguage(TreeLanguage[Expr]):
    rules = (
        TreeRule("simplify", simplify),
        TreeRule("demorgan", demorgan),
        TreeRule(
            "commute",
            lambda t: replace(t, args=tuple(reversed(t.args))) if len(t.args) == 2 else None,
        ),
        TreeRule("associate_left", lambda t: associate(t, True)),
        TreeRule("associate_right", lambda t: associate(t, False)),
    )

    def children(self, term: Expr) -> tuple[Expr, ...]:
        return term.args

    def rebuild(self, term: Expr, children: tuple[Expr, ...]) -> Expr:
        return replace(term, args=children)

    def node_label(self, term: Expr) -> str:
        # Only the absorbing/identity masks matter to the rules, at any width.
        if term.op != "const":
            return term.op
        if term.value == 0:
            return "const:0"
        return "const:ones" if term.value == term.mask else "const:other"
