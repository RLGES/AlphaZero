"""Learn-R's built-in rewrite rules (egraph_bridge/egg_egraph.py) as patterns.

Each egglog ``rewrite(lhs).to(rhs)`` becomes one search action. Bidirectional
commutativity is one action (it is its own inverse); associativity becomes two
actions, one per direction. ``fold_constants`` evaluates an operator whose
operands are all constants; Learn-R's egglog rules have no folding, but its
SSA e-graph pipeline does, and compiled programs need it.
"""

from __future__ import annotations

from benchmarks.common import TreeRule

from .patterns import pattern_rule
from .term import Const, Expr, apply_op

# (name, lhs, rhs) in the order of Learn-R's create_*_rules functions.
ALGEBRAIC = (
    ("add_zero_r", "(add ?x 0)", "?x"),
    ("add_zero_l", "(add 0 ?x)", "?x"),
    ("sub_zero", "(sub ?x 0)", "?x"),
    ("sub_self", "(sub ?x ?x)", "0"),
    ("mul_one_r", "(mul ?x 1)", "?x"),
    ("mul_one_l", "(mul 1 ?x)", "?x"),
    ("mul_zero_r", "(mul ?x 0)", "0"),
    ("mul_zero_l", "(mul 0 ?x)", "0"),
    ("and_self", "(and ?x ?x)", "?x"),
    ("or_self", "(or ?x ?x)", "?x"),
    ("xor_self", "(xor ?x ?x)", "0"),
    ("xor_zero_r", "(xor ?x 0)", "?x"),
    ("xor_zero_l", "(xor 0 ?x)", "?x"),
    ("and_zero_r", "(and ?x 0)", "0"),
    ("and_zero_l", "(and 0 ?x)", "0"),
    ("or_zero_r", "(or ?x 0)", "?x"),
    ("or_zero_l", "(or 0 ?x)", "?x"),
    ("shl_zero", "(shl ?x 0)", "?x"),
    ("shr_zero", "(shr ?x 0)", "?x"),
    ("zero_shl", "(shl 0 ?x)", "0"),
    ("zero_shr", "(shr 0 ?x)", "0"),
    ("neg_neg", "(neg (neg ?x))", "?x"),
    ("not_not", "(not (not ?x))", "?x"),
    ("phi_same", "(phi ?x ?x)", "?x"),
)

STRENGTH_REDUCTION = (
    ("mul2_shl", "(mul ?x 2)", "(shl ?x 1)"),
    ("mul4_shl", "(mul ?x 4)", "(shl ?x 2)"),
    ("mul8_shl", "(mul ?x 8)", "(shl ?x 3)"),
    ("mul16_shl", "(mul ?x 16)", "(shl ?x 4)"),
    ("div2_shr", "(div ?x 2)", "(shr ?x 1)"),
    ("div4_shr", "(div ?x 4)", "(shr ?x 2)"),
    ("add_self_shl", "(add ?x ?x)", "(shl ?x 1)"),
)

COMMUTATIVITY = tuple(
    (f"commute_{op}", f"({op} ?x ?y)", f"({op} ?y ?x)") for op in ("add", "mul", "and", "or", "xor")
)

ASSOCIATIVITY = tuple(
    rule
    for op in ("add", "mul", "and", "or")
    for rule in (
        (f"assoc_{op}_right", f"({op} ({op} ?x ?y) ?z)", f"({op} ?x ({op} ?y ?z))"),
        (f"assoc_{op}_left", f"({op} ?x ({op} ?y ?z))", f"({op} ({op} ?x ?y) ?z)"),
    )
)

PATTERN_RULES = ALGEBRAIC + STRENGTH_REDUCTION + COMMUTATIVITY + ASSOCIATIVITY


def _fold(term: Expr) -> Expr | None:
    if term.args and all(arg.op == "const" for arg in term.args) and term.op != "load":
        if term.op == "phi":
            return None
        return Const(apply_op(term.op, [int(arg.value) for arg in term.args]))
    return None


FOLD = TreeRule("fold_constants", _fold)
BUILTIN_RULES: tuple[TreeRule[Expr], ...] = (
    FOLD,
    *(pattern_rule(name, lhs, rhs) for name, lhs, rhs in PATTERN_RULES),
)
