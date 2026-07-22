"""FilmOps: Operator Toolkit for FilmBench.

Six specialized operators for classifying professional cinematographic
attributes: shot scale, composition, camera angle, color & tone, character
layout, and camera movement.

Typical usage::

    from filmops import FilmOpsConfig, FilmOpsPipeline

    cfg = FilmOpsConfig(checkpoint_dir="./checkpoints")
    pipe = FilmOpsPipeline(cfg)
    pipe.load()
    result = pipe.analyse_video("shot.mp4")
"""

from filmops._version import __version__
from filmops.config import FilmOpsConfig
from filmops.core.base import BaseOperator
from filmops.core.exceptions import (
    FilmOpsError,
    OperatorInferenceError,
    OperatorLoadError,
    OperatorNotFoundError,
)
from filmops.core.pipeline import FilmOpsPipeline
from filmops.core.registry import list_operators, register_operator

# Triggers each operator module to self-register with the registry.
from filmops import operators as _operators  # noqa: F401

__all__ = [
    "FilmOpsConfig",
    "FilmOpsPipeline",
    "BaseOperator",
    "register_operator",
    "list_operators",
    "FilmOpsError",
    "OperatorLoadError",
    "OperatorInferenceError",
    "OperatorNotFoundError",
    "__version__",
]
