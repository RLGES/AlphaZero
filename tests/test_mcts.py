from dataclasses import dataclass
from math import inf, nan

import pytest

from src import Engine, EngineConfig, Language, MCTSConfig, RewriteRule, rewrite, rewrite_detailed
from src.engines.mcts import MCTSEngine


class GraphLanguage(Language[int, RewriteRule]):
    """No tree or hashable-term assumptions are needed for the engine."""

    def __init__(self, edges, costs):
        self.edges = edges
        self.costs = costs

    def applicable(self, term):
        return tuple(RewriteRule(str(i)) for i, _ in enumerate(self.edges.get(term, ())))

    def apply(self, term, action):
        return self.edges[term][int(action.name)]

    def cost(self, term):
        return self.costs[term]

    def fingerprint(self, term):
        return term


def test_public_api_and_best_intermediate_rollout():
    # The cheapest state occurs inside a rollout; its terminal endpoint is worse.
    language = GraphLanguage({0: (1,), 1: (2,), 2: (3,)}, {0: 10, 1: 8, 2: 1, 3: 30})
    config = MCTSConfig(iterations=1, rollout_depth=10)
    result = rewrite_detailed(language, 0, config)
    assert rewrite(language, 0, config) == 2
    assert result.initial_cost == 10
    assert result.final_cost == 1
    assert result.statistics["best_iteration"] == 1
    assert [step.after for step in result.trace] == [1, 2]
    assert result.trace[-1].after == result.final_term
    assert result.wall_time >= 0


def test_crosses_a_temporary_cost_increase():
    language = GraphLanguage({0: (1,), 1: (2,)}, {0: 10, 1: 20, 2: 1})
    result = rewrite_detailed(language, 0, MCTSConfig(iterations=1))
    assert result.final_term == 2
    assert result.trace[0].after_cost > result.initial_cost


def test_cycles_are_path_local_and_bounded():
    language = GraphLanguage({0: (1,), 1: (0,)}, {0: 10, 1: 9})
    result = rewrite_detailed(language, 0, MCTSConfig(iterations=20, rollout_depth=1000))
    assert result.final_term == 1
    assert result.statistics["cycles_detected"] == 20
    assert result.statistics["states_explored"] == 2
    assert result.statistics["transitions"] == 2
    assert result.statistics["rollout_steps"] <= 20


def test_transpositions_share_states_but_preserve_valid_trace():
    language = GraphLanguage(
        {0: (1, 2), 1: (3,), 2: (3,), 3: (4,)}, {0: 10, 1: 9, 2: 8, 3: 7, 4: 1}
    )
    result = rewrite_detailed(language, 0, MCTSConfig(iterations=40, rollout_depth=4, seed=4))
    assert result.final_term == 4
    assert result.statistics["states_explored"] == 5
    assert result.statistics["transpositions"] >= 1
    state = result.initial_term
    for step in result.trace:
        assert step.before == state
        state = language.apply(state, step.action)
        assert step.after == state
    assert state == result.final_term


def test_seed_is_reproducible_and_engine_can_be_reused():
    language = GraphLanguage({0: (1, 2), 1: (3, 0), 2: (4, 0)}, {0: 10, 1: 12, 2: 9, 3: 2, 4: 1})
    engine = MCTSEngine(MCTSConfig(iterations=30, seed=17))
    first, second = engine.run(language, 0), engine.run(language, 0)
    assert first.final_term == second.final_term
    assert first.statistics == second.statistics
    assert first.trace == second.trace


def test_zero_budget_and_terminal_input():
    language = GraphLanguage({}, {0: 0})
    result = rewrite_detailed(language, 0, MCTSConfig(iterations=0))
    assert result.final_term == 0
    assert result.improvement_percent == 0
    assert not result.trace
    assert result.statistics["iterations"] == 0
    assert rewrite(language, 0) == 0


def test_state_budget_stops_a_growing_graph():
    language = GraphLanguage({i: (i + 1,) for i in range(20)}, {i: 20 - i for i in range(21)})
    result = rewrite_detailed(language, 0, MCTSConfig(iterations=100, max_states=3))
    assert result.statistics["states_explored"] == 3
    assert result.statistics["stop_reason"] == "max_states"
    assert result.final_term == 2


def test_time_budget(monkeypatch):
    from src.engines.mcts import engine as module

    counter = iter(i * 0.001 for i in range(100))
    monkeypatch.setattr(module, "perf_counter", lambda: next(counter))
    language = GraphLanguage({0: (1,), 1: (0,)}, {0: 10, 1: 1})
    result = rewrite_detailed(language, 0, MCTSConfig(iterations=1000, time_limit=0.005))
    assert result.statistics["stop_reason"] == "time_limit"
    assert result.statistics["iterations"] < 1000


def test_custom_engine_dispatch():
    class IdentityEngine(Engine):
        def run(self, language, term):
            return MCTSEngine(MCTSConfig(iterations=0)).run(language, term)

    @dataclass(frozen=True)
    class IdentityConfig(EngineConfig):
        def create_engine(self):
            return IdentityEngine()

    language = GraphLanguage({}, {0: 3})
    assert rewrite(language, 0, IdentityConfig()) == 0


def test_terms_need_not_be_hashable():
    class ListLanguage(Language[list[int], int]):
        def applicable(self, term):
            return (0,) if len(term) > 1 else ()

        def apply(self, term, action):
            return term[:1]

        def cost(self, term):
            return len(term)

        def fingerprint(self, term):
            return tuple(term)

    assert rewrite(ListLanguage(), [1, 2, 3], MCTSConfig(iterations=2)) == [1]


@pytest.mark.parametrize("cost", [-1, inf, nan])
def test_invalid_language_costs_fail_clearly(cost):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rewrite(GraphLanguage({}, {0: cost}), 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": -1},
        {"iterations": 1.5},
        {"iterations": True},
        {"rollout_depth": -1},
        {"max_tree_depth": 0},
        {"max_states": 0},
        {"exploration_constant": -1},
        {"exploration_constant": inf},
        {"time_limit": 0},
        {"time_limit": nan},
    ],
)
def test_invalid_mcts_configs(kwargs):
    with pytest.raises(ValueError):
        MCTSConfig(**kwargs)
