"""Tests for the config dataclasses."""

from filmops import FilmOpsConfig
from filmops.config import OperatorConfigs


def test_default_config():
    cfg = FilmOpsConfig()
    assert cfg.checkpoint_dir == "./checkpoints"
    assert cfg.video_fps == 5
    assert cfg.batch_size == 8
    assert isinstance(cfg.operator_configs, OperatorConfigs)


def test_is_operator_enabled_default_all():
    cfg = FilmOpsConfig()
    assert cfg.is_operator_enabled("shot_scale")
    assert cfg.is_operator_enabled("anything")


def test_is_operator_enabled_subset():
    cfg = FilmOpsConfig(enabled_operators=["shot_scale"])
    assert cfg.is_operator_enabled("shot_scale")
    assert not cfg.is_operator_enabled("composition")


def test_load_kwargs_resolves_paths():
    cfg = FilmOpsConfig(checkpoint_dir="/tmp/ck")
    kwargs = cfg.operator_configs.load_kwargs(
        "shot_scale", base_checkpoint_dir=cfg.checkpoint_dir, device="cpu",
    )
    assert kwargs["device"] == "cpu"
    assert kwargs["model_path"].startswith("/tmp/ck/shot_scale/")
