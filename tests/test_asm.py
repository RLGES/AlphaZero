from collections import Counter
from random import Random

import pytest

from benchmarks.asm.patterns import build, match, parse_pattern, pattern_rule, verify_rule
from benchmarks.asm.term import MASK, Const, Expr, Var, evaluate, from_sexpr, to_sexpr

x, y = Var("x"), Var("y")


def B(op, a, b):
    return Expr(op, (a, b))


def test_32bit_semantics():
    env = {"x": 7, "y": 0}
    assert evaluate(B("sub", Const(0), Const(1)), env) == MASK
    assert evaluate(B("mul", Const(1 << 31), Const(2)), env) == 0
    assert evaluate(B("div", x, y), env) == 0 and evaluate(B("mod", x, y), env) == 0
    assert evaluate(B("shl", Const(1), Const(33)), env) == 2  # count masked to 5 bits
    assert evaluate(Expr("not", (Const(0),)), env) == MASK
    assert evaluate(Expr("neg", (Const(1),)), env) == MASK
    assert evaluate(B("phi", x, x), env) == 7
    assert evaluate(Expr("load", (x,)), env) == evaluate(Expr("load", (Const(7),)), env)
    assert Const(-1) == Const(MASK)


def test_invalid_terms_rejected():
    with pytest.raises(ValueError):
        Expr("add", (x,))
    with pytest.raises(ValueError):
        Expr("frobnicate", (x, y))


def test_sexpr_round_trip_and_format():
    term = B("add", B("mul", x, Const(2)), Expr("not", (Const(-3),)))
    assert from_sexpr(to_sexpr(term)) == term
    assert to_sexpr(term) == "(add (mul x 2) (not -3))"
    assert str(term) == "((x * 2) + ~-3)"


def test_pattern_match_nonlinear_and_build():
    lhs = parse_pattern("(sub ?x ?x)")
    assert match(lhs, B("sub", x, x)) == {"x": x}
    assert match(lhs, B("sub", x, y)) is None
    assert match(parse_pattern("(mul ?x 2)"), B("mul", y, Const(2))) == {"x": y}
    assert build(parse_pattern("(shl ?x 1)"), {"x": y}) == B("shl", y, Const(1))


def test_pattern_rule_rewrites_first_match_top_down():
    rule = pattern_rule("add_zero", "(add ?x 0)", "?x")
    assert rule.transform(B("add", x, Const(0))) == x
    assert rule.transform(B("add", x, Const(1))) is None


def test_verify_rule_accepts_sound_and_rejects_unsound():
    rng = Random(0)
    assert verify_rule("(mul ?x 2)", "(shl ?x 1)", rng)
    assert verify_rule("(div ?x 4)", "(shr ?x 2)", rng)
    assert not verify_rule("(mul ?a ?b)", "?a", rng)  # Learn-R's "MUL t4, t3 -> t4"
    assert not verify_rule("(add ?x 1)", "?y", rng)  # rhs variable not bound by lhs


def test_learned_rule_generalisation_and_rejections():
    from benchmarks.asm.learned import UnsupportedRule, generalize, parse_llm_text

    rng = Random(0)
    lhs, rhs = generalize(["MOV t, x", "SHL t, 1", "ADD t, x"], ["MOV r, x", "IMUL r, 3"], rng)
    assert (lhs, rhs) == ("(add (shl ?x 1) ?x)", "(mul ?x 3)")
    for left, right, reason in (
        (["MUL t4, t3"], ["mov eax, t4"], "unsound"),
        (["MOV t2, a", "ADD t2, b"], ["mov eax, a", "add eax, b"], "no-op"),
        (["CMP x, y", "JGE L0"], ["mov eax, 1"], "unsupported"),
    ):
        with pytest.raises(UnsupportedRule, match=reason):
            generalize(left, right, rng)
    text = "LHS:\n  MOV t, x   ; copy\n  XOR t, t\nRHS:\n  mov t, 0\nSaturating E-Graph...\n"
    text += "  LHS: ['MUL a, 1']\n  RHS: ['mov a, a']\n"
    assert parse_llm_text(text) == [
        (["MOV t, x", "XOR t, t"], ["mov t, 0"]),
        (["MUL a, 1"], ["mov a, a"]),
    ]


def test_import_writes_report_and_language_loads_it(tmp_path):
    import json

    from benchmarks.asm.learned import import_rules, load_learned_rules

    source = tmp_path / "db.json"
    rules = [
        {"lhs_seq": ["MOV t, x", "SHL t, 1", "ADD t, x"], "rhs_seq": ["MOV r, x", "IMUL r, 3"]},
        {"lhs_seq": ["MUL t4, t3"], "rhs_seq": ["mov eax, t4"]},
    ]
    source.write_text(json.dumps({"rules": rules}))
    out = tmp_path / "learned.json"
    report = import_rules([source], builtin=set(), out=out)
    assert [r["lhs"] for r in report["accepted"]] == ["(add (shl ?x 1) ?x)"]
    assert len(report["rejected"]) == 1
    (rule,) = load_learned_rules(out)
    assert rule.name == "learned_0"
    assert rule.transform(Expr("add", (Expr("shl", (x, Const(1))), x))) == Expr(
        "mul", (x, Const(3))
    )


def test_every_builtin_rule_is_sound():
    from benchmarks.asm.rules import PATTERN_RULES

    rng = Random(7)
    assert len(PATTERN_RULES) == 44
    for name, lhs, rhs in PATTERN_RULES:
        assert verify_rule(lhs, rhs, rng), name


def test_fold_constants_and_weighted_cost():
    from benchmarks.asm.language import AsmLanguage
    from benchmarks.asm.rules import FOLD

    language = AsmLanguage()
    assert FOLD.transform(B("add", Const(2), Const(3))) == Const(5)
    assert FOLD.transform(B("add", x, Const(3))) is None
    assert FOLD.transform(B("phi", Const(1), Const(2))) is None
    assert language.cost(B("mul", x, Const(2))) == 5  # mul 3 + two leaves
    assert language.cost(B("shl", x, Const(1))) == 3  # strength reduction pays off
    assert language.cost(B("div", x, Const(4))) > language.cost(B("shr", x, Const(2)))


def test_ir_inlining_matches_learnr_lowering():
    from benchmarks.asm.learnr_bridge import ir_to_terms

    ir = ["t0 = 5", "a = t0", "t1 = 0", "t2 = a + t1", "t3 = in0 * t2", "b = t3"]
    terms, straight_line = ir_to_terms(ir)
    assert straight_line
    assert terms == {"a": Const(5), "b": B("mul", Var("in0"), B("add", Const(5), Const(0)))}
    _, straight_line = ir_to_terms(["CMP a, b", "JGE L0", "x = a"])
    assert not straight_line


def test_greedy_misses_what_needs_a_reassociation(monkeypatch):
    from benchmarks.asm.hardness import greedy_cost, hardness, is_hard
    from benchmarks.asm.language import BUILTIN_RULES, AsmLanguage

    # Built-in rules only: a learned rule (RLGES_ASM_LEARNED) may fold x*2*4 in one step.
    monkeypatch.setattr(AsmLanguage, "rules", BUILTIN_RULES)
    language = AsmLanguage()
    term = B("mul", B("mul", x, Const(2)), Const(4))  # x*2*4: best is x<<3 (cost 3)
    assert language.cost(term) == 9
    assert greedy_cost(language, term) == 5  # shl(shl(x, 1), 2): no cheaper single step
    stats = hardness(term)
    assert stats["strong"] == 3 and is_hard(stats)
    assert not is_hard(hardness(B("add", x, Const(0))))


def test_hard_style_programs_compile():
    from benchmarks.asm.learnr_bridge import DEFAULT_LEARNR, program_terms, random_program

    if not (DEFAULT_LEARNR / "frontend").is_dir():
        pytest.skip("no Learn-R checkout")
    rng = Random(3)
    classic = random_program(Random(3))
    assert random_program(Random(3)) == classic  # classic stays reproducible
    terms = [
        t
        for _ in range(20)
        for t in program_terms(random_program(rng, style="hard"), DEFAULT_LEARNR, 150)
    ]
    assert len(terms) > 40


def test_corpus_splits_are_disjoint_and_generator_uses_them():
    from benchmarks.asm.generator import corpus, random_term
    from benchmarks.generators import generate

    train, test = corpus("train"), corpus("test")
    assert len(train) > 1000 and len(test) > 100
    assert not set(train) & set(test)
    held_out = generate("asm", Random(5), 40, split="test")
    assert held_out == generate("asm", Random(5), 40, split="test")
    assert any(term in set(test) for term in held_out)
    assert not any(term in set(train) for term in held_out if term in set(test))
    assert isinstance(random_term(Random(0), 3), Expr)


def test_rules_preserve_semantics_on_corpus_terms():
    from benchmarks.asm.generator import corpus
    from benchmarks.asm.language import AsmLanguage

    language, rng = AsmLanguage(), Random(11)
    for term in rng.sample(corpus("train"), 300):
        names = {n.value for n in _nodes(term) if n.op == "var"}
        env = {name: rng.getrandbits(32) for name in names}
        expected = evaluate(term, env)
        for rule in language.applicable(term):
            assert evaluate(language.apply(term, rule), env) == expected, rule.name


def _nodes(term):
    yield term
    for arg in term.args:
        yield from _nodes(arg)


def test_asm_benchmarks_are_optimizable():
    from benchmarks.registry import load_suite
    from src import MCTSConfig, rewrite_detailed

    suite = load_suite("asm")
    names = {b.name for b in suite.benchmarks}
    assert {"learnr_test_code", "learnr_test_code3", "shift_fold"} <= names
    for benchmark in suite.benchmarks:
        result = rewrite_detailed(suite.language, benchmark.term, MCTSConfig(iterations=300))
        assert result.final_cost < result.initial_cost, benchmark.name


def test_z3_proves_sound_rules_and_refutes_unsound_ones():
    pytest.importorskip("z3")
    from benchmarks.asm.rules import PATTERN_RULES
    from benchmarks.asm.smt import prove_rule

    assert all(prove_rule(lhs, rhs)[0] == "proved" for _, lhs, rhs in PATTERN_RULES)
    assert prove_rule("(sub (add ?x ?y) ?y)", "?x")[0] == "proved"
    assert prove_rule("(add ?x ?y)", "(sub ?x ?y)")[0] == "counterexample"
    assert prove_rule("(div (mul ?x ?y) ?y)", "?x")[0] == "counterexample"  # y = 0, overflow
    assert prove_rule("(mod ?x ?x)", "0")[0] == "proved"  # x % 0 is 0 too


def test_llm_text_keeps_list_items_with_comments():
    from benchmarks.asm.learned import parse_llm_text

    text = "LHS: ['MOV t1, a ; copy', 'ADD t1, 0']\nRHS: ['MOV t1, a']\n"
    assert parse_llm_text(text) == [(["MOV t1, a", "ADD t1, 0"], ["MOV t1, a"])]


def test_discovery_accepts_only_new_proved_rules():
    pytest.importorskip("z3")
    from benchmarks.asm.discover import discover, parse_proposals

    reply = "\n".join(
        [
            "Sure! Here are rules:",
            '{"name": "cancel", "lhs": "(sub (add ?x ?y) ?y)", "rhs": "?x"}',
            '{"name": "again", "lhs": "(sub (add ?a ?b) ?b)", "rhs": "?a"}',
            '{"name": "known", "lhs": "(add ?x 0)", "rhs": "?x"}',
            '{"name": "bad", "lhs": "(add ?x ?y)", "rhs": "(sub ?x ?y)"}',
            '{"name": "trap", "lhs": "(div (mul ?x ?y) ?y)", "rhs": "?x"}',
            '{"name": "free", "lhs": "(add ?x 1)", "rhs": "?z"}',
            '{"name": "reg", "lhs": "(cmp eax 0)", "rhs": "0"}',
            "not json {",
        ]
    )
    assert len(parse_proposals(reply)) == 7
    seeds = [B("sub", B("add", x, y), y)]
    report = discover(lambda prompt, t: reply, seeds, prompts=2, workers=1, log=lambda _: None)
    assert [(r["lhs"], r["rhs"]) for r in report["accepted"]] == [("(sub (add ?x ?y) ?y)", "?x")]
    assert report["accepted"][0]["seed_matches"] == 1 and report["accepted"][0]["lowers_cost"]
    reasons = Counter(r["reason"] for r in report["rejected"])
    assert reasons["duplicate"] == 5  # renamed copy + built-in, then all three again
    assert reasons["unsound (random 32-bit test)"] == 4
    assert reasons["right side uses unbound variables"] == 2
    assert reasons["unparseable"] == 2  # bare register names are not pattern variables


def test_learned_rules_selectable_by_environment(monkeypatch, tmp_path):
    from benchmarks.asm.learned import LEARNED_PATH, learned_path

    monkeypatch.delenv("RLGES_ASM_LEARNED", raising=False)
    assert learned_path() == LEARNED_PATH
    monkeypatch.setenv("RLGES_ASM_LEARNED", "none")
    assert learned_path() is None
    monkeypatch.setenv("RLGES_ASM_LEARNED", str(tmp_path / "r.json"))
    assert learned_path() == tmp_path / "r.json"


def test_control_flow_splits_into_exact_basic_blocks():
    from benchmarks.asm.learnr_bridge import ir_to_block_terms

    ir = ["t0 = 0", "i = t0", "L0:", "t1 = 3", "CMP i, t1", "JGE L1", "t2 = 2", "t3 = i * t2"]
    ir += ["t4 = s + t3", "s = t4", "t5 = 1", "t6 = i + t5", "i = t6", "JMP L0", "L1:"]
    blocks = ir_to_block_terms(ir)
    assert blocks[0] == {"i": Const(0)}
    assert blocks[1] == {"cmp0": Var("i"), "cmp1": Const(3)}
    s, i = Var("s"), Var("i")
    assert blocks[2] == {"s": B("add", s, B("mul", i, Const(2))), "i": B("add", i, Const(1))}


def test_rank_rules_loads_only_cost_lowering_rules():
    from benchmarks.asm.discover import rank_rules

    def rule(lhs, rhs, lhs_cost, rhs_cost, seed_matches):
        return {
            "name": lhs,
            "lhs": lhs,
            "rhs": rhs,
            "lhs_cost": lhs_cost,
            "rhs_cost": rhs_cost,
            "seed_matches": seed_matches,
        }

    verified = [
        rule("(shl ?x 1)", "(mul ?x 2)", 3, 5, 100),
        rule("(mod ?x 4)", "(and ?x 3)", 12, 3, 2),
        rule("(sub (add ?x ?y) ?z)", "(add ?x (sub ?y ?z))", 5, 5, 6),
        rule("(div ?x 1)", "?x", 12, 1, 2),
    ]
    accepted, rest = rank_rules(verified, max_accepted=1)
    assert [r["lhs"] for r in accepted] == ["(div ?x 1)"]
    assert accepted[0]["name"].startswith("llm_0_")
    assert [r["lhs"] for r in rest] == ["(shl ?x 1)", "(sub (add ?x ?y) ?z)", "(mod ?x 4)"]


def test_final_eval_sets_and_rule_guard(tmp_path, monkeypatch):
    import json

    from benchmarks.asm import final_eval

    def write(name, sexprs):
        (tmp_path / name).write_text("".join(json.dumps({"sexpr": s}) + "\n" for s in sexprs))

    write("train.jsonl", ["(add r0 0)"])
    write("test.jsonl", ["(mul r1 1)", "(mul r1 1)"])
    write("cf_test.jsonl", ["(sub r2 r2)"])
    write("stress_test.jsonl", ["(add r0 0)", "(mul r1 1)", "(xor r3 r3)"])
    monkeypatch.setattr(final_eval, "CORPUS_DIR", tmp_path)
    sets = final_eval.test_sets()
    assert [name for name, _ in sets["heldout"]] == ["heldout/0000", "heldout/0001"]
    assert len(sets["stress"]) == 1  # the two corpus terms are dropped

    checkpoints = tmp_path / "checkpoints"
    run = checkpoints / "asm_v3_llm_s0"
    run.mkdir(parents=True)
    (checkpoints / "summary_asm_v3_llm_s0.txt").write_text(
        "Overnight summary: asm_v3_llm_s0, rules some/rules.json (2026-10-25 09:02:49)\n"
    )
    monkeypatch.setattr(final_eval, "CHECKPOINTS", checkpoints)
    monkeypatch.setenv("RLGES_ASM_LEARNED", "none")
    with pytest.raises(SystemExit, match="trained with rules"):
        final_eval.evaluate_runs([run], ["heldout"], (10,), workers=1, log=lambda _: None)


def test_random_program_length_override():
    from benchmarks.asm.learnr_bridge import random_program

    source = random_program(Random(3), 20, style="hard")
    assert len([line for line in source.splitlines() if " = " in line]) >= 20


def test_merge_rules_prefixes_later_rounds_and_skips_repeats():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "scripts" / "merge_rules.py"
    spec = importlib.util.spec_from_file_location("merge_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r1 = {"accepted": [{"name": "llm_0_a", "lhs": "(a)", "rhs": "b"}]}
    r2 = {
        "accepted": [
            {"name": "llm_0_a", "lhs": "(a)", "rhs": "b"},
            {"name": "llm_1_c", "lhs": "(c)", "rhs": "d"},
        ]
    }
    merged = module.merge([r1, r2])
    assert [r["name"] for r in merged] == ["llm_0_a", "r2_llm_1_c"]


def test_egglog_adapter_is_sound_and_folds_32_bit():
    pytest.importorskip("egglog")
    from random import Random

    from benchmarks.asm.language import AsmLanguage
    from benchmarks.asm.term import Expr, evaluate
    from src.api import rewrite_detailed
    from src.engines.egglog import EgglogConfig

    language = AsmLanguage()
    x, y = Var("x"), Var("y")
    cases = [
        Expr("sub", (Expr("mul", (x, y)), Expr("mul", (y, x)))),  # x*y - y*x -> 0
        Expr("mul", (Const(0xFFFF_FFFF), Const(0xFFFF_FFFF))),  # folds to 1 (mod 2^32)
        Expr("add", (Expr("shl", (x, Const(33))), Expr("div", (y, Const(0))))),
        Expr("mod", (Expr("neg", (Const(5),)), Const(7))),
    ]
    rng = Random(0)
    for term in cases:
        result = rewrite_detailed(language, term, EgglogConfig())
        assert result.final_cost <= result.initial_cost
        for _ in range(50):
            env = {"x": rng.getrandbits(32), "y": rng.getrandbits(32)}
            assert evaluate(result.final_term, env) == evaluate(term, env)
    assert rewrite_detailed(language, cases[0], EgglogConfig()).final_term == Const(0)
    assert rewrite_detailed(language, cases[1], EgglogConfig()).final_term == Const(1)


def test_egglog_table_pairs_alphazero_with_egglog(tmp_path, monkeypatch, capsys):
    import csv
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "scripts" / "egglog_table.py"
    spec = importlib.util.spec_from_file_location("egglog_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints" / "run_s0").mkdir(parents=True)
    fields = ["term", "engine", "budget", "improvement"]

    def write(name, rows):
        with open(name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dict(zip(fields, row)) for row in rows)

    write(
        "checkpoints/final_egglog_none_heldout.csv",
        [("a", "egglog", 0, 90), ("b", "egglog", 0, 50)],
    )
    write(
        "checkpoints/run_s0/final_heldout.csv",
        [
            ("a", "AlphaZero", 10, 80),
            ("b", "AlphaZero", 10, 60),
            ("a", "MCTS", 10, 70),
            ("b", "MCTS", 10, 40),
        ],
    )
    assert module.main(["--set", "heldout", "arm=none:checkpoints/run_s*"]) == 0
    row = capsys.readouterr().out.splitlines()[-1]
    assert row.startswith("| arm | 1 | 10 | 70.0 | 55.0 | 70.0 | — | — | +0.0")
    write(
        "checkpoints/final_egglogmcts_none_heldout.csv",
        [("a", "egglog+MCTS", 10, 90), ("b", "egglog+MCTS", 10, 55)],
    )
    write(
        "checkpoints/run_s0/final_heldout_hybrid.csv",
        [("a", "egglog+AlphaZero", 10, 92), ("b", "egglog+AlphaZero", 10, 60)],
    )
    assert module.main(["--set", "heldout", "arm=none:checkpoints/run_s*"]) == 0
    row = capsys.readouterr().out.splitlines()[-1]
    assert "| 72.5 | 76.0 |" in row
    assert row.endswith("| +6.0 [+2.0, +10.0] | +3.5 [+2.0, +5.0] |")


def test_final_eval_hybrid_never_worse_than_egglog():
    pytest.importorskip("egglog")
    from benchmarks.asm import final_eval
    from benchmarks.asm.term import Expr

    x, y = Var("x"), Var("y")
    term = Expr("add", (Expr("mul", (x, Const(8))), Expr("sub", (y, y))))
    (egglog_row,) = final_eval._evaluate(("t", term, "egglog", (5,)))
    rows = final_eval._evaluate(("t", term, "egglog+mcts", (5, 10)))
    assert [r["engine"] for r in rows] == ["egglog+MCTS", "egglog+MCTS"]
    for row in rows:
        assert row["initial_cost"] == egglog_row["initial_cost"]
        assert row["final_cost"] <= egglog_row["final_cost"]
