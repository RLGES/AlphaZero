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


def test_heldout_score_best_checkpoint_and_early_stop(tmp_path):
    out = tmp_path / "v2"
    args = [*TINY, "--depth", "1,2", "--heldout", "3", "--patience", "1", "--out", str(out)]
    assert main([*args, "--rounds", "6"]) == 0
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    for row in rows:
        assert 0.0 <= row["heldout_mean_improvement"] <= 1.0
        assert row["score"] == row["heldout_mean_improvement"]
    assert len(rows) == 6 or rows[-1]["rounds_since_best"] >= 1
    best_round = max(rows, key=lambda r: r["score"])["round"]
    torch = pytest.importorskip("torch")
    assert torch.load(out / "best.pt", weights_only=False)["round"] == best_round
    # Resuming keeps the best score instead of restarting the comparison.
    assert main([*args, "--rounds", str(len(rows) + 1), "--resume", "--patience", "0"]) == 0
    resumed = json.loads((out / "metrics.jsonl").read_text().splitlines()[-1])
    assert resumed["best_score"] >= max(r["score"] for r in rows)


@pytest.mark.parametrize("depth", ["5,2", "3", "a,b"])
def test_bad_depth_rejected(tmp_path, depth):
    with pytest.raises(SystemExit):
        main([*TINY, "--rounds", "1", "--depth", depth, "--out", str(tmp_path / "x")])


def test_resume_without_checkpoint_fails(tmp_path):
    with pytest.raises(SystemExit):
        main([*TINY, "--rounds", "1", "--out", str(tmp_path / "none"), "--resume"])
