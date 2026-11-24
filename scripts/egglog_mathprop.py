"""egglog baselines on the MCTS-GEB benchmarks (their Math and Prop languages and rules).

    PYTHONPATH=. python scripts/egglog_mathprop.py <exprs.json> <out.csv> [node_limit]

Expressions come from MCTS-GEB's own generator (exported by /root/rlges/external/export_exprs.py).
Rules are transcribed one-to-one from their pes/interface/MathLang.py (26) and PropLang.py
(13). No constant folding: their benchmark runs build the e-graph without an evaluation
function (pes/utils/utils.py new_egraph), so their egg does none either. Cost = AstSize.
The node limit is strict: an iteration that would cross it is rolled back (push/pop), so
egglog never uses more e-nodes than the limit (MCTS-GEB's environment stops at the limit).
Schedules, each run until the e-node limit, saturation or 30 iterations:
  egglog-default: back-off scheduler, match limit 1000, ban length 2 (our egglog default)
  BackOff-200:    back-off scheduler, match limit 200, ban length 2
"""

from __future__ import annotations

import csv
import json
import sys
import time

from egglog import (EGraph, Expr, String, back_off, i64, rewrite, rule, ruleset, run,
                    union, vars_)


class M(Expr):
    def __init__(self, v: i64) -> None: ...
    @classmethod
    def sym(cls, s: String) -> M: ...
    def __add__(self, o: M) -> M: ...
    def __sub__(self, o: M) -> M: ...
    def __mul__(self, o: M) -> M: ...
    def __truediv__(self, o: M) -> M: ...
    def __pow__(self, o: M) -> M: ...
    @classmethod
    def diff(cls, x: M, y: M) -> M: ...
    @classmethod
    def integral(cls, x: M, y: M) -> M: ...
    @classmethod
    def sqrt(cls, x: M) -> M: ...
    @classmethod
    def sin(cls, x: M) -> M: ...
    @classmethod
    def cos(cls, x: M) -> M: ...


class P(Expr):
    @classmethod
    def lit(cls, v: i64) -> P: ...  # 1 = True, 0 = False
    @classmethod
    def sym(cls, s: String) -> P: ...
    def and_(self, o: P) -> P: ...
    def or_(self, o: P) -> P: ...
    def implies(self, o: P) -> P: ...
    def not_(self) -> P: ...


def _math_forms():
    p, q = vars_("p q", M)
    k = list(vars_("k", i64))[0]
    s = list(vars_("s", String))[0]
    return [M(k), M.sym(s), p + q, p - q, p * q, p / q, p ** q, M.diff(p, q),
            M.integral(p, q), M.sqrt(p), M.sin(p), M.cos(p)]


def _prop_forms():
    p, q = vars_("p q", P)
    k = list(vars_("k", i64))[0]
    s = list(vars_("s", String))[0]
    return [P.lit(k), P.sym(s), p.and_(q), p.or_(q), p.implies(q), p.not_()]


def math_rules():
    a, b, c, x, f, g = vars_("a b c x f g", M)
    i, j = vars_("i j", i64)
    return ruleset(
        rewrite(a + b).to(b + a),                                   # comm-add
        rewrite(a * b).to(b * a),                                   # comm-mul
        rewrite((a + b) + c).to(a + (b + c)),                       # assoc-add
        rewrite((a * b) * c).to(a * (b * c)),                       # assoc-mul
        rewrite(a - b).to(a + M(-1) * b),                           # sub-canon
        rewrite(a + M(0)).to(a),                                    # zero-add
        rewrite(a * M(0)).to(M(0)),                                 # zero-mul
        rewrite(a * M(1)).to(a),                                    # one-mul
        # add-zero (a -> a + 0) and mul-one (a -> a * 1) match every e-class in egg; egglog
        # needs a grounded left side, so they are stated once per constructor (same effect).
        *[rewrite(t).to(t + M(0)) for t in _math_forms()],
        *[rewrite(t).to(t * M(1)) for t in _math_forms()],
        rewrite(a - a).to(M(0)),                                    # cancel-sub
        rewrite(a * (b + c)).to(a * b + a * c),                     # distribute
        rewrite(a * b + a * c).to(a * (b + c)),                     # factor
        rewrite(a ** b * a ** c).to(a ** (b + c)),                  # pow-mul
        rewrite(x ** M(1)).to(x),                                   # pow1
        rewrite(x ** M(2)).to(x * x),                               # pow2
        rewrite(M.diff(x, a + b)).to(M.diff(x, a) + M.diff(x, b)),  # d-add
        rewrite(M.diff(x, a * b)).to(a * M.diff(x, b) + b * M.diff(x, a)),  # d-mul
        rewrite(M.diff(x, M.sin(x))).to(M.cos(x)),                  # d-sin
        rewrite(M.diff(x, M.cos(x))).to(M(-1) * M.sin(x)),          # d-cos
        rewrite(M.integral(M(1), x)).to(x),                         # i-one
        rewrite(M.integral(M.cos(x), x)).to(M.sin(x)),              # i-cos
        rewrite(M.integral(M.sin(x), x)).to(M(-1) * M.cos(x)),      # i-sin
        rewrite(M.integral(f + g, x)).to(M.integral(f, x) + M.integral(g, x)),  # i-sum
        rewrite(M.integral(f - g, x)).to(M.integral(f, x) - M.integral(g, x)),  # i-dif
        rewrite(M.integral(a * b, x)).to(                                       # i-parts
            a * M.integral(b, x) - M.integral(M.diff(x, a) * M.integral(b, x), x)),
        name="math",
    )


def prop_rules():
    a, b, c = vars_("a b c", P)
    i, j = vars_("i j", i64)
    T = P.lit(1)
    return ruleset(
        rewrite(a.implies(b)).to(a.not_().or_(b)),                  # def_imply
        rewrite(a.not_().not_()).to(a),                             # double_neg
        rewrite(a.not_().or_(b)).to(a.implies(b)),                  # def_imply_flip
        *[rewrite(t).to(t.not_().not_()) for t in _prop_forms()],  # double_neg_flip, per form
        rewrite(a.or_(b.or_(c))).to(a.or_(b).or_(c)),               # assoc_or
        rewrite(a.and_(b.or_(c))).to(a.and_(b).or_(a.and_(c))),     # dist_and_or
        rewrite(a.or_(b.and_(c))).to(a.or_(b).and_(a.or_(c))),      # dist_or_and
        rewrite(a.or_(b)).to(b.or_(a)),                             # comm_or
        rewrite(a.and_(b)).to(b.and_(a)),                           # comm_and
        rewrite(a.or_(a.not_())).to(T),                             # lem
        rewrite(a.or_(T)).to(T),                                    # or_true
        rewrite(a.and_(T)).to(a),                                   # and_true
        rewrite(a.implies(b)).to(b.not_().implies(a.not_())),       # contrapositive
        name="prop",
    )


MATH_OPS = {"Add": "__add__", "Sub": "__sub__", "Mul": "__mul__", "Div": "__truediv__",
            "Pow": "__pow__"}


def encode(lang: str, t):
    if lang == "MATH":
        if isinstance(t, list):
            op, *args = t
            enc = [encode(lang, a) for a in args]
            if op in MATH_OPS:
                return getattr(enc[0], MATH_OPS[op])(enc[1])
            return getattr(M, op.lower())(*enc)
        return M(int(t)) if not isinstance(t, str) else M.sym(t)
    if isinstance(t, list):
        op, *args = t
        enc = [encode(lang, a) for a in args]
        if op == "Not":
            return enc[0].not_()
        return {"And": P.and_, "Or": P.or_, "Implies": P.implies}[op](enc[0], enc[1])
    if isinstance(t, bool):
        return P.lit(int(t))
    return P.sym(str(t))


def ast_size(t) -> int:
    return 1 + sum(ast_size(a) for a in t[1:]) if isinstance(t, list) else 1


def ast_cost(egraph, expr, child_costs: list[int]) -> int:
    """egg's AstSize: every expression node costs 1; primitive payloads (i64, String) cost 0."""
    if isinstance(expr, (M, P)):
        return 1 + sum(child_costs)
    return 0


def _size(egraph) -> int:
    return sum(n for _, n in egraph.all_function_sizes())


def solve(lang: str, tree, limit: int, match_limit: int):
    egraph = EGraph()
    root = encode(lang, tree)
    egraph.register(root)
    rules = math_rules() if lang == "MATH" else prop_rules()
    schedule = run(rules, scheduler=back_off(match_limit=match_limit, ban_length=2))
    start = time.perf_counter()
    for _ in range(30):
        before = _size(egraph)
        egraph.push()
        egraph.run(schedule)
        after = _size(egraph)
        if after > limit:
            egraph.pop()  # roll back the iteration that crossed the limit
            break
        if after == before:
            break
    _, cost = egraph.extract(root, include_cost=True, cost_model=ast_cost)
    return int(cost), _size(egraph), time.perf_counter() - start


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    rows = []
    for item in json.load(open(src)):
        row = {"lang": item["lang"], "seed": item["seed"], "limit": limit,
               "base_cost": ast_size(item["tree"])}
        for name, ml in (("egglog_default", 1000), ("backoff200", 200)):
            cost, size, sec = solve(item["lang"], item["tree"], limit, ml)
            row.update({f"{name}_cost": cost, f"{name}_enodes": size, f"{name}_s": round(sec, 3)})
        rows.append(row)
        print(row, flush=True)
    with open(dst, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
