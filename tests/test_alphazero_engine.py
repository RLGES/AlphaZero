from random import Random

import pytest

from benchmarks.arithmetic import ArithmeticLanguage, Var
from benchmarks.registry import lookup
from src import AlphaZeroConfig, Language, RewriteRule, rewrite_detailed
from src.api.cli import main
from src.engines.alphazero.features import GraphFeatures
from src.engines.alphazero.search import PUCTSearch, UniformEvaluator
from src.engines.egglog.engine import UnsupportedLanguageError


class FeaturedGraphLanguage(Language[int, RewriteRule]):
    """Explicit state graph whose action i is the i-th outgoing edge."""

    def __init__(self, edges, costs):
        self.edges = edges
        self.costs = costs
        width = max((len(v) for v in edges.values()), default=0)
        self.vocabulary = tuple(RewriteRule(str(i)) for i in range(width))

    def applicable(self, term):
        return self.vocabulary[: len(self.edges.get(term, ()))]

    def apply(self, term, action):
        return self.edges[term][int(action.name)]

    def cost(self, term):
        return self.costs[term]

    def fingerprint(self, term):
        return term

    def features(self, term):
        return GraphFeatures((f"s{term}",), ())

    def action_vocabulary(self):
        return self.vocabulary


FACTOR_SUITE, FACTOR = lookup("arithmetic/factor_once")


def test_untrained_engine_finds_factor_once_deterministically():
    config = AlphaZeroConfig(iterations=100, seed=0)
    first = rewrite_detailed(FACTOR_SUITE.language, FACTOR.term, config)
    second = rewrite_detailed(FACTOR_SUITE.language, FACTOR.term, config)
    assert first.engine == "AlphaZero"
    assert first.final_cost == 5.0
    assert first.final_term == second.final_term
    assert first.statistics == second.statistics
    assert first.trace and first.trace[-1].after == first.final_term


def test_no_actions_returns_input_without_evaluations():
    result = rewrite_detailed(ArithmeticLanguage(), Var("x"), AlphaZeroConfig(iterations=10))
    assert result.final_term == Var("x")
    assert result.statistics["network_evaluations"] == 0


def test_zero_iterations_returns_input():
    result = rewrite_detailed(FACTOR_SUITE.language, FACTOR.term, AlphaZeroConfig(iterations=0))
    assert result.final_term == FACTOR.term and result.trace == ()


def test_cycles_terminate():
    language = FeaturedGraphLanguage({0: (1,), 1: (0,)}, {0: 10, 1: 9})
    result = rewrite_detailed(language, 0, AlphaZeroConfig(iterations=20))
    assert result.final_term == 1
    assert result.statistics["iterations"] == 20
    assert result.statistics["states_explored"] == 2


def test_crosses_temporary_cost_increase():
    language = FeaturedGraphLanguage({0: (1,), 1: (2,)}, {0: 10, 1: 20, 2: 1})
    result = rewrite_detailed(language, 0, AlphaZeroConfig(iterations=5))
    assert result.final_term == 2


def test_zero_initial_cost_is_safe():
    language = FeaturedGraphLanguage({0: (1,)}, {0: 0, 1: 0})
    result = rewrite_detailed(language, 0, AlphaZeroConfig(iterations=5))
    assert result.final_cost == 0


def test_visits_cover_only_applicable_actions():
    language = FACTOR_SUITE.language
    search = PUCTSearch(language, UniformEvaluator(), AlphaZeroConfig(iterations=50), Random(0))
    result = search.run(FACTOR.term)
    applicable = set(language.applicable(FACTOR.term))
    assert sum(result.visits) == 50
    for action, visits in zip(language.action_vocabulary(), result.visits):
        if action not in applicable:
            assert visits == 0


def test_dirichlet_noise_is_seeded():
    language = FACTOR_SUITE.language
    config = AlphaZeroConfig(iterations=30, dirichlet_epsilon=0.25)
    runs = [PUCTSearch(language, UniformEvaluator(), config, Random(3)).run(FACTOR.term)]
    runs.append(PUCTSearch(language, UniformEvaluator(), config, Random(3)).run(FACTOR.term))
    assert runs[0].visits == runs[1].visits


def test_unsupported_language():
    class Plain(FeaturedGraphLanguage):
        features = None

    with pytest.raises(UnsupportedLanguageError):
        rewrite_detailed(Plain({}, {0: 1}), 0, AlphaZeroConfig())


@pytest.mark.parametrize(
    "kwargs",
    [{"iterations": -1}, {"c_puct": -1.0}, {"dirichlet_epsilon": 2.0}, {"max_states": 0}],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        AlphaZeroConfig(**kwargs)


def test_cli_engine_flag(capsys):
    assert main(["arithmetic/factor_once", "--engine", "alphazero", "--iterations", "50"]) == 0
    assert "Engine: AlphaZero" in capsys.readouterr().out
