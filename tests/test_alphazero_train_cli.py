import json

import pytest

pytest.importorskip("torch")

from src.engines.alphazero.train import main  # noqa: E402

TINY = ["--suite", "arithmetic", "--games", "3", "--steps", "3", "--sims", "4"]
TINY += ["--batch-size", "4", "--eval-iterations", "5", "--hidden", "16", "--layers", "1"]


def test_train_one_round_then_resume(tmp_path, capsys):
    out = str(tmp_path / "run")
    assert main([*TINY, "--rounds", "1", "--out", out]) == 0
    assert (tmp_path / "run" / "latest.pt").exists()
    assert (tmp_path / "run" / "theta_k0000.pt").exists()
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    assert first["round"] == 0
    for key in ("loss", "policy_loss", "value_loss", "samples", "eval_mean_final_cost"):
        assert key in first

    assert main([*TINY, "--rounds", "2", "--out", out, "--resume"]) == 0
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["round"] for line in lines] == [0, 1]
    assert (tmp_path / "run" / "theta_k0001.pt").exists()
    assert "round 1" in capsys.readouterr().out


def test_resume_without_checkpoint_fails(tmp_path):
    with pytest.raises(SystemExit):
        main([*TINY, "--rounds", "1", "--out", str(tmp_path / "none"), "--resume"])
