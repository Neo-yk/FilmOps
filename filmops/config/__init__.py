"""FilmOps configuration.

Two sub-modules with clearly separated responsibilities:

* :mod:`pipeline_config` — runtime parameters (device, batch_size, ...).
* :mod:`operator_configs` — per-operator checkpoint / weight paths.
"""

from filmops.config.operator_configs import (
    CameraAngleConfig,
    CameraMovementConfig,
    CharacterLayoutConfig,
    ColorToneConfig,
    CompositionConfig,
    OperatorConfigs,
    ShotScaleConfig,
)
from filmops.config.pipeline_config import FilmOpsConfig

__all__ = [
    "FilmOpsConfig",
    "OperatorConfigs",
    "ShotScaleConfig",
    "CompositionConfig",
    "CameraAngleConfig",
    "ColorToneConfig",
    "CharacterLayoutConfig",
    "CameraMovementConfig",
]
