import pytest

pytest.importorskip("modal")
pytest.importorskip("torch")

import modal_app  # noqa: E402


def test_app_registers_training_functions():
    assert modal_app.app.name == "rewritezero-alphazero"
    for name in ("selfplay_chunk", "train_driver"):
        assert isinstance(getattr(modal_app, name), modal_app.modal.Function)


def test_default_options_are_the_v2_recipe():
    options = modal_app.default_options(suite="boolean")
    cfg = modal_app.train_config(options)
    assert options["run_name"] == "boolean_v2"
    assert cfg.out_dir == f"{modal_app.CHECKPOINT_ROOT}/boolean_v2"
    assert cfg.spec.num_actions == 7 and cfg.spec.hidden == 192
    assert cfg.depth == (3, 7) and cfg.heldout_terms == 30 and cfg.patience == 8
    assert cfg.selfplay.simulations_per_move == 128 and cfg.selfplay.max_moves == 40


def test_bad_depth_fails_before_launch():
    with pytest.raises(Exception, match="MIN"):
        modal_app.train_config(modal_app.default_options(depth="7,3"))
