import pytest

from benchmarks.arithmetic import Add, Const, Mul, Var
from benchmarks.arithmetic.language import ArithmeticLanguage
from benchmarks.arithmetic.term import evaluate
from benchmarks.boolean.term import evaluate as boolean_eval
from benchmarks.registry import load_suite
from src import EgglogConfig, rewrite, rewrite_detailed
from src.engines.egglog import UnsupportedLanguageError


@pytest.mark.parametrize("suite_name", ["arithmetic", "boolean"])
def test_supported_egglog_comparison(suite_name):
    pytest.importorskip("egglog")
    suite = load_suite(suite_name)
    benchmark = suite.benchmarks[0]
    result = rewrite_detailed(suite.language, benchmark.term, EgglogConfig(iterations=4))
    assert result.engine == "egglog"
    assert result.final_cost < result.initial_cost
    assert result.statistics["extraction_cost"] == result.final_cost
    assert result.wall_time >= 0
    assert result.statistics["rule_matches"] > 0
    assert not result.trace


def test_egglog_arithmetic_uses_unbounded_integers():
    pytest.importorskip("egglog")
    language = ArithmeticLanguage()
    huge = 2**100
    term = Mul(Add(Const(huge), Const(huge)), Const(3))
    result = rewrite_detailed(language, term, EgglogConfig(iterations=4))
    assert result.final_term == Const(6 * huge)
    assert evaluate(result.final_term, {}) == evaluate(term, {})
    assert result.statistics["extraction_cost"] == 1


def test_egglog_boolean_constant_folding_and_semantics():
    pytest.importorskip("egglog")
    from benchmarks.boolean import And, Bool, Not, Or

    suite = load_suite("boolean")
    term = Not(And(Bool(True), Bool(False)))
    assert rewrite(suite.language, term, EgglogConfig()) == Bool(True)
    for benchmark in suite.benchmarks:
        result = rewrite_detailed(suite.language, benchmark.term, EgglogConfig(iterations=4))
        for p in (False, True):
            for q in (False, True):
                env = {"p": p, "q": q, "r": False}
                assert boolean_eval(benchmark.term, env) == boolean_eval(result.final_term, env)
    assert rewrite(suite.language, Or(Bool(False), Bool(True)), EgglogConfig()) == Bool(True)


def test_egglog_zero_iterations_and_leaf_cost():
    pytest.importorskip("egglog")
    term = Add(Var("x"), Const(0))
    result = rewrite_detailed(ArithmeticLanguage(), term, EgglogConfig(iterations=0))
    assert result.final_term == term
    assert result.statistics["extraction_cost"] == 3


def test_unsupported_suite_is_explicit():
    suite = load_suite("lambda")
    with pytest.raises(UnsupportedLanguageError, match="MCTS only"):
        rewrite(suite.language, suite.benchmarks[0].term, EgglogConfig())


@pytest.mark.parametrize(
    "kwargs", [{"iterations": -1}, {"iterations": 1.5}, {"match_limit": 0}, {"match_limit": True}]
)
def test_invalid_egglog_config(kwargs):
    with pytest.raises(ValueError):
        EgglogConfig(**kwargs)
