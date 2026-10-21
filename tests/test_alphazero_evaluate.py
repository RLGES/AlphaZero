import csv

import pytest

from src.engines.alphazero.evaluate import main


def test_tiny_evaluation_prints_all_engines(tmp_path, capsys):
    out = tmp_path / "rows.csv"
    args = ["--suite", "arithmetic", "--budgets", "5", "--seeds", "1", "--generated", "2"]
    assert main([*args, "--csv", str(out)]) == 0
    text = capsys.readouterr().out
    for word in ("MCTS", "AlphaZero", "egglog", "transitions", "best-known"):
        assert word in text
    rows = list(csv.DictReader(out.open()))
    assert {row["engine"] for row in rows} == {"MCTS", "AlphaZero", "egglog"}
    assert len({row["term"] for row in rows}) == 5 + 2


def test_checkpoint_requires_torch_or_runs(tmp_path):
    pytest.importorskip("torch")
    from benchmarks.arithmetic import ArithmeticLanguage
    from src.engines.alphazero.checkpoint import save_checkpoint
    from src.engines.alphazero.network import NetworkSpec, PolicyValueNet

    path = tmp_path / "c.pt"
    save_checkpoint(path, PolicyValueNet(NetworkSpec(6, 64, 16, 1)), ArithmeticLanguage(), 0, {})
    args = ["--suite", "arithmetic", "--budgets", "3", "--seeds", "1", "--checkpoint", str(path)]
    assert main([*args, "--no-egglog"]) == 0
