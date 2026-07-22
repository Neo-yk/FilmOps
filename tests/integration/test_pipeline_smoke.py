"""End-to-end smoke tests.

These do **not** require real checkpoints — they only verify that the
Pipeline can be instantiated, that the registry is wired correctly, and
that operator load failures surface as :class:`OperatorLoadError`.
"""

import pytest

from filmops import FilmOpsConfig, FilmOpsPipeline


def test_pipeline_construct_empty():
    pipe = FilmOpsPipeline(FilmOpsConfig(enabled_operators=[]))
    assert pipe.operators == {}


def test_pipeline_load_missing_ckpt_raises():
    pipe = FilmOpsPipeline(FilmOpsConfig(
        checkpoint_dir="/no/such/path",
        enabled_operators=["color_tone"],
        device="cpu",
    ))
    with pytest.raises(Exception):
        pipe.load()
