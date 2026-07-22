"""Operator implementations.

Importing this package triggers each operator module to self-register via
the :func:`filmops.core.register_operator` decorator. The Pipeline can
then discover them by name without any hard-coded dispatch table.
"""

# Side-effect imports: register every operator with the registry.
from filmops.operators.camera_angle import CameraAngleOperator  # noqa: F401
from filmops.operators.camera_movement import CameraMovementOperator  # noqa: F401
from filmops.operators.character_layout import CharacterLayoutOperator  # noqa: F401
from filmops.operators.color_tone import ColorToneOperator  # noqa: F401
from filmops.operators.composition import CompositionOperator  # noqa: F401
from filmops.operators.shot_scale import ShotScaleOperator  # noqa: F401

__all__ = [
    "CameraAngleOperator",
    "CameraMovementOperator",
    "CharacterLayoutOperator",
    "ColorToneOperator",
    "CompositionOperator",
    "ShotScaleOperator",
]
