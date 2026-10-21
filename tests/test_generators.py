from random import Random

import pytest

from benchmarks.generators import GENERATOR_SUITES, generate
from benchmarks.registry import load_suite
from src import MCTSConfig, rewrite_detailed


@pytest.mark.parametrize("suite_name", GENERATOR_SUITES)
def test_generator_deterministic_valid_and_held_out(suite_name):
    first = generate(suite_name, Random(1), 50)
    assert first == generate(suite_name, Random(1), 50)
    assert len(first) == len(set(first)) == 50
    suite = load_suite(suite_name)
    language = suite.language
    held_out = {benchmark.term for benchmark in suite.benchmarks}
    assert all(term not in held_out and language.applicable(term) for term in first)
    improvable = sum(
        rewrite_detailed(language, term, MCTSConfig(iterations=20, rollout_depth=4)).final_cost
        < language.cost(term)
        for term in first
    )
    assert improvable >= 25


@pytest.mark.parametrize("suite_name", GENERATOR_SUITES)
def test_depth_bounds_size(suite_name):
    language = load_suite(suite_name).language
    terms = generate(suite_name, Random(2), 20, depth=(1, 2))
    # Each grammar level may add one planted wrapper level, so depth <= 2 * 2 + 1.
    assert max(language.size(term) for term in terms) < 2 ** (2 * 2 + 2)


def test_unknown_suite():
    with pytest.raises(ValueError, match="lambda"):
        generate("lambda", Random(0), 1)
