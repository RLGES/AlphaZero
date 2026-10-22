from dataclasses import replace

from benchmarks.common import TreeLanguage, TreeRule

from .learned import learned_path, load_learned_rules
from .rules import BUILTIN_RULES
from .term import Expr

# Rough instruction latencies: the cost model makes strength reduction pay off.
OP_COST = {"mul": 3, "div": 10, "mod": 10, "load": 4}


class AsmLanguage(TreeLanguage[Expr]):
    """Learn-R's Asm IR: built-in rules plus verified learned rules as actions."""

    rules: tuple[TreeRule[Expr], ...] = BUILTIN_RULES + load_learned_rules(learned_path())

    def children(self, term: Expr) -> tuple[Expr, ...]:
        return term.args

    def rebuild(self, term: Expr, children: tuple[Expr, ...]) -> Expr:
        return replace(term, args=children)

    def cost(self, term: Expr) -> float:
        total = 0.0
        pending = [term]
        while pending:
            node = pending.pop()
            total += OP_COST.get(node.op, 1)
            pending.extend(node.args)
        return total

    def node_label(self, term: Expr) -> str:
        if term.op == "var":
            return "var"
        if term.op != "const":
            return term.op
        value = term.signed
        if -16 <= value <= 16:
            return f"const:{value}"
        return "const:pow2" if value > 0 and value & (value - 1) == 0 else "const:big"
