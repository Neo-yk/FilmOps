"""Smoke tests for the operator registry."""

import pytest

import filmops
from filmops.core.exceptions import OperatorNotFoundError
from filmops.core.registry import list_operators, resolve_operator


EXPECTED = {
    "shot_scale", "composition", "camera_angle",
    "color_tone", "character_layout", "camera_movement",
}


def test_all_operators_registered():
    names = set(list_operators())
    missing = EXPECTED - names
    assert not missing, f"Missing from registry: {missing}"


def test_resolve_known_operator():
    cls = resolve_operator("shot_scale")
    assert cls.name == "shot_scale"


def test_resolve_unknown_operator_raises():
    with pytest.raises(OperatorNotFoundError):
        resolve_operator("nonexistent_op")


def test_public_api_exports():
    for attr in ("FilmOpsConfig", "FilmOpsPipeline", "BaseOperator",
                 "register_operator", "list_operators", "__version__"):
        assert hasattr(filmops, attr), f"Missing top-level export: {attr}"
