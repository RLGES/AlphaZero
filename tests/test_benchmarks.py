from importlib import import_module
from itertools import product

import pytest

from benchmarks.arithmetic import Add as AAdd
from benchmarks.arithmetic import Const as AConst
from benchmarks.arithmetic import Mul
from benchmarks.arithmetic import Var as AVar
from benchmarks.arithmetic.language import ArithmeticLanguage
from benchmarks.arithmetic.term import evaluate as arithmetic_eval
from benchmarks.bitvector import Const as BVConst
from benchmarks.bitvector import Var as BVVar
from benchmarks.bitvector import Xor
from benchmarks.bitvector.term import evaluate as bitvector_eval
from benchmarks.boolean.term import evaluate as boolean_eval
from benchmarks.linear_algebra import Add as MAdd
from benchmarks.linear_algebra import Identity, MatMul, Matrix, Transpose, Zero
from benchmarks.linear_algebra.language import LinearAlgebraLanguage
from benchmarks.linear_algebra.term import evaluate as matrix_eval
from benchmarks.regex import Alt, Eps, Lit, Seq, Star, matches
from benchmarks.registry import SUITE_NAMES, all_benchmarks, load_suite, lookup
from src import MCTSConfig, rewrite_detailed


def reachable(language, term, depth=2):
    frontier, seen = [term], {term}
    for _ in range(depth):
        following = []
        for current in frontier:
            for action in language.applicable(current):
                after = language.apply(current, action)
                assert after != current
                if after not in seen:
                    seen.add(after)
                    following.append(after)
        frontier = following
    return seen


@pytest.mark.parametrize(
    "suite,benchmark", all_benchmarks(), ids=[f"{s.name}/{b.name}" for s, b in all_benchmarks()]
)
def test_every_benchmark_smoke(suite, benchmark):
    result = rewrite_detailed(
        suite.language, benchmark.term, MCTSConfig(iterations=30, rollout_depth=6, seed=0)
    )
    assert result.final_cost <= result.initial_cost
    state = benchmark.term
    for step in result.trace:
        assert step.before == state
        state = suite.language.apply(state, step.action)
        assert state == step.after
    assert state == result.final_term


def test_registry_and_difficulty_metadata():
    assert len(SUITE_NAMES) == 6
    for name in SUITE_NAMES:
        suite = load_suite(name)
        assert suite.name == name
        assert len(suite.benchmarks) >= 3
        assert {"easy", "medium", "hard"} <= {b.difficulty for b in suite.benchmarks}
        for benchmark in suite.benchmarks:
            assert lookup(f"{name}/{benchmark.name}") == (suite, benchmark)
            assert benchmark.description
    with pytest.raises(ValueError, match="Unknown suite"):
        load_suite("missing")
    with pytest.raises(ValueError, match="Unknown benchmark"):
        lookup("arithmetic/missing")
    with pytest.raises(ValueError, match="form"):
        lookup("arithmetic")


def test_batch_rewrites_siblings_and_blocks_overlapping_matches():
    language = ArithmeticLanguage()
    rule = language.rules[0]
    x, y = AVar("x"), AVar("y")
    term = AAdd(AAdd(x, AConst(0)), Mul(y, AConst(1)))
    assert language.apply(term, rule) == AAdd(x, y)
    nested = AAdd(AConst(0), AAdd(x, AConst(0)))
    assert language.apply(nested, rule) == AAdd(x, AConst(0))
    assert language.apply(language.apply(nested, rule), rule) == x


def test_expansion_limit_is_a_language_policy():
    language = ArithmeticLanguage(max_nodes=6)
    term = Mul(AVar("x"), AAdd(AVar("y"), AVar("z")))
    assert "distribute" in {a.name for a in ArithmeticLanguage().applicable(term)}
    assert "distribute" not in {a.name for a in language.applicable(term)}


def test_arithmetic_rules_preserve_exact_integer_semantics():
    suite = load_suite("arithmetic")
    for benchmark in suite.benchmarks:
        for env_values in product((-2, 0, 3), repeat=3):
            env = dict(zip(("x", "y", "z"), env_values))
            expected = arithmetic_eval(benchmark.term, env)
            for term in reachable(suite.language, benchmark.term):
                assert arithmetic_eval(term, env) == expected


def test_boolean_rules_preserve_truth_tables():
    suite = load_suite("boolean")
    for benchmark in suite.benchmarks:
        for values in product((False, True), repeat=3):
            env = dict(zip(("p", "q", "r"), values))
            expected = boolean_eval(benchmark.term, env)
            for term in reachable(suite.language, benchmark.term):
                assert boolean_eval(term, env) == expected


def test_bitvector_rules_and_wraparound():
    suite = load_suite("bitvector")
    terms = [b.term for b in suite.benchmarks] + [
        Xor(Xor(BVVar("x", 3), BVVar("y", 3)), BVVar("x", 3))
    ]
    for original in terms:
        for x, y in product(range(8), repeat=2):
            env = {"x": x, "y": y}
            expected = bitvector_eval(original, env)
            for term in reachable(suite.language, original):
                assert bitvector_eval(term, env) == expected
    assert BVConst(-1, 8).value == 255
    assert BVConst(260, 8).value == 4
    with pytest.raises(ValueError, match="width"):
        Xor(BVVar("x", 8), BVVar("y", 16))
    with pytest.raises(ValueError):
        BVVar("x", 0)


def test_lambda_capture_avoidance_shadowing_and_eta_side_condition():
    terms = import_module("benchmarks.lambda.term")
    language = load_suite("lambda").language
    x, y = terms.Var("x"), terms.Var("y")
    result = terms.substitute(terms.Abs("y", x), "x", y)
    assert result.parameter != "y"
    assert result.body == y
    assert terms.free_vars(result) == {"y"}
    assert terms.substitute(terms.Abs("x", x), "x", y) == terms.Abs("x", x)
    nested = terms.Abs("y", terms.Abs("y_0", x))
    result = terms.substitute(nested, "x", y)
    assert result.parameter not in ("y", "y_0")
    assert terms.free_vars(result) == {"y"}
    bad_eta = terms.Abs("x", terms.App(x, x))
    assert "eta" not in {a.name for a in language.applicable(bad_eta)}
    omega_abs = terms.Abs("x", terms.App(x, x))
    omega = terms.App(omega_abs, omega_abs)
    assert not language.applicable(omega)


def test_matrix_rules_preserve_values_and_shapes():
    language = LinearAlgebraLanguage()
    p, q, r = (Matrix(name, 2, 2) for name in ("P", "Q", "R"))
    env = {"P": ((1, 2), (3, 4)), "Q": ((2, 0), (1, -1)), "R": ((0, 3), (1, 2))}
    terms = [
        MatMul(Identity(2), p),
        MAdd(MatMul(p, q), MatMul(p, r)),
        Transpose(MatMul(MatMul(p, q), r)),
        MAdd(MatMul(p, Zero(2, 2)), Transpose(Transpose(q))),
        MatMul(MAdd(p, q), r),
    ]
    for original in terms:
        expected = matrix_eval(original, env)
        for term in reachable(language, original):
            assert (term.rows, term.cols) == (original.rows, original.cols)
            assert matrix_eval(term, env) == expected
    a, b, c = Matrix("A", 64, 8), Matrix("B", 8, 256), Matrix("C", 256, 2)
    assert language.cost(MatMul(a, MatMul(b, c))) < language.cost(MatMul(MatMul(a, b), c))
    with pytest.raises(ValueError, match="dimensions"):
        MatMul(a, Matrix("bad", 3, 4))
    with pytest.raises(ValueError, match="dimensions"):
        MAdd(a, b)
    assert matrix_eval(MatMul(p, q), env) != matrix_eval(MatMul(q, p), env)


def test_regex_rules_preserve_languages_on_bounded_strings():
    suite = load_suite("regex")
    strings = ["".join(chars) for length in range(4) for chars in product("abc", repeat=length)]
    for benchmark in suite.benchmarks:
        for term in reachable(suite.language, benchmark.term):
            assert [matches(term, text) for text in strings] == [
                matches(benchmark.term, text) for text in strings
            ]
    nullable = Star(Alt(Eps(), Lit("a")))
    assert matches(nullable, "") and matches(nullable, "aaa")
    assert not matches(nullable, "b")
    assert matches(Seq(Star(Lit("a")), Lit("b")), "aaab")
