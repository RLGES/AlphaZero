import pytest

from src.api.cli import main


def test_cli_single_benchmark_and_trace(capsys):
    assert main(["arithmetic/factor_once", "--iterations", "20", "--seed", "0", "--verbose"]) == 0
    output = capsys.readouterr().out
    for label in (
        "Benchmark: arithmetic/factor_once",
        "Difficulty: easy",
        "Initial cost:",
        "Final cost:",
        "Improvement:",
        "Time:",
        "Iterations:",
        "States seen:",
        "Best found:",
        "Rewrite trace:",
        "factor",
    ):
        assert label in output


@pytest.mark.parametrize("args", [["suite", "boolean"], ["all"]])
def test_cli_aggregates(args, capsys):
    assert main([*args, "--iterations", "2"]) == 0
    output = capsys.readouterr().out
    assert "Benchmark" in output and "Improvement" in output
    assert "boolean/demorgan" in output
    if args == ["all"]:
        assert "regex/epsilon" in output and "lambda/nested_beta" in output


def test_unsupported_egglog_is_explained(capsys):
    assert main(["lambda/nested_beta", "--iterations", "2", "--compare-egglog"]) == 0
    output = capsys.readouterr().out
    assert "MCTS" in output
    assert "comparison skipped" in output
    assert "MCTS only" in output


def test_missing_optional_dependency_is_explained(monkeypatch, capsys):
    import src.api.cli as cli
    from src.engines.egglog import EgglogUnavailableError

    actual = cli.rewrite_detailed

    def missing(language, term, config):
        if isinstance(config, cli.EgglogConfig):
            raise EgglogUnavailableError("Install it with: pip install -e '.[egglog]'")
        return actual(language, term, config)

    monkeypatch.setattr(cli, "rewrite_detailed", missing)
    assert main(["arithmetic/factor_once", "--iterations", "1", "--compare-egglog"]) == 0
    assert "Install it with" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["missing/name"],
        ["arithmetic/missing"],
        ["arithmetic"],
        ["suite"],
        ["all", "boolean"],
        ["all", "--iterations", "-1"],
    ],
)
def test_cli_lookup_errors_are_clean(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_listing(capsys):
    assert main(["list"]) == 0
    assert "arithmetic/factor_once" in capsys.readouterr().out
