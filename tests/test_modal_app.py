import pytest

pytest.importorskip("modal")
pytest.importorskip("torch")

import modal_app  # noqa: E402


def test_app_registers_training_functions():
    assert modal_app.app.name == "rewritezero-alphazero"
    for name in ("selfplay_chunk", "train_driver"):
        assert isinstance(getattr(modal_app, name), modal_app.modal.Function)


def test_options_build_a_valid_train_config():
    options = modal_app.default_options(suite="boolean", run_name="b1", rounds=2)
    cfg = modal_app.train_config(options)
    assert cfg.suite == "boolean" and cfg.rounds == 2
    assert cfg.out_dir == f"{modal_app.CHECKPOINT_ROOT}/b1"
    assert cfg.spec.num_actions == 7
