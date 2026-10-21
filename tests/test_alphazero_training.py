import pickle
from random import Random

import pytest

from benchmarks.arithmetic import ArithmeticLanguage, Var
from benchmarks.boolean import BooleanLanguage
from benchmarks.generators import generate
from src import Language, RewriteRule
from src.engines.alphazero.features import GraphFeatures
from src.engines.alphazero.search import UniformEvaluator
from src.engines.alphazero.selfplay import ReplayBuffer, SelfPlayConfig, play_episode

torch = pytest.importorskip("torch")

from src.engines.alphazero.checkpoint import load_network, save_checkpoint  # noqa: E402
from src.engines.alphazero.network import NetworkSpec, PolicyValueNet  # noqa: E402
from src.engines.alphazero.training import evaluate_losses, train_steps  # noqa: E402


class ChainLanguage(Language[int, RewriteRule]):
    """0 -> 1 -> 2 with a temporary cost increase."""

    vocabulary = (RewriteRule("next"),)
    costs = {0: 10.0, 1: 20.0, 2: 1.0}

    def applicable(self, term):
        return self.vocabulary if term < 2 else ()

    def apply(self, term, action):
        return term + 1

    def cost(self, term):
        return self.costs[term]

    def fingerprint(self, term):
        return term

    def features(self, term):
        return GraphFeatures((f"s{term}",), ())

    def action_vocabulary(self):
        return self.vocabulary


def test_episode_targets_use_best_future_cost():
    samples = play_episode(
        ChainLanguage(), 0, UniformEvaluator(), SelfPlayConfig(simulations_per_move=4), Random(0)
    )
    assert [s.value for s in samples] == pytest.approx([0.9, 0.95])
    assert all(s.policy == (1.0,) and s.mask == (True,) for s in samples)
    assert pickle.loads(pickle.dumps(samples)) == samples


def test_episode_on_arithmetic_is_well_formed():
    language = ArithmeticLanguage()
    term = generate("arithmetic", Random(0), 1, depth=(2, 3))[0]
    samples = play_episode(
        language, term, UniformEvaluator(), SelfPlayConfig(simulations_per_move=16), Random(0)
    )
    assert samples
    for sample in samples:
        assert abs(sum(sample.policy) - 1) < 1e-9
        assert all(p == 0 for p, legal in zip(sample.policy, sample.mask) if not legal)
        assert 0.0 <= sample.value <= 1.0


def test_no_action_term_yields_no_samples():
    language = ArithmeticLanguage()
    assert play_episode(language, Var("x"), UniformEvaluator(), SelfPlayConfig(), Random(0)) == []


def test_replay_buffer_capacity_and_sampling():
    buffer = ReplayBuffer(capacity=3)
    buffer.extend([1, 2, 3, 4])
    assert len(buffer) == 3
    assert set(buffer.sample(10, Random(0))) <= {2, 3, 4}


def _samples(count: int = 6):
    language = ArithmeticLanguage()
    samples = []
    for term in generate("arithmetic", Random(1), count, depth=(2, 3)):
        samples += play_episode(
            language, term, UniformEvaluator(), SelfPlayConfig(simulations_per_move=8), Random(0)
        )
    return samples


def test_training_reduces_loss_on_fixed_buffer():
    torch.manual_seed(0)
    samples = _samples()
    buffer = ReplayBuffer(1000)
    buffer.extend(samples)
    net = PolicyValueNet(NetworkSpec(num_actions=6, buckets=128, hidden=32, layers=2))
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3)
    before = evaluate_losses(net, samples)
    stats = train_steps(net, optimizer, buffer, steps=150, batch_size=16, rng=Random(0))
    after = evaluate_losses(net, samples)
    assert set(stats) == {"loss", "policy_loss", "value_loss"}
    assert after["loss"] < before["loss"]
    assert after["value_loss"] < 0.5 * before["value_loss"]


def test_checkpoint_round_trip_and_mismatch(tmp_path):
    net = PolicyValueNet(NetworkSpec(num_actions=6, buckets=64, hidden=16, layers=1))
    path = tmp_path / "c.pt"
    save_checkpoint(path, net, ArithmeticLanguage(), round_index=3, metrics={"loss": 1.0})
    loaded = load_network(path, ArithmeticLanguage(), "cpu")
    assert loaded.spec == net.spec
    for a, b in zip(net.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="rule"):
        load_network(path, BooleanLanguage(), "cpu")


def test_engine_runs_from_checkpoint(tmp_path):
    from src import AlphaZeroConfig, rewrite_detailed

    language = ArithmeticLanguage()
    net = PolicyValueNet(NetworkSpec(num_actions=6, buckets=64, hidden=16, layers=1))
    save_checkpoint(tmp_path / "c.pt", net, language, round_index=0, metrics={})
    term = generate("arithmetic", Random(4), 1, depth=(2, 3))[0]
    config = AlphaZeroConfig(iterations=30, checkpoint=str(tmp_path / "c.pt"))
    result = rewrite_detailed(language, term, config)
    assert result.statistics["network_evaluations"] > 0
    assert result.final_cost <= result.initial_cost
