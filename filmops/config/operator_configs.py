"""Per-operator dataclass configurations.

Each operator's checkpoint / weight paths live in their own dataclass.
The :class:`OperatorConfigs` container holds one instance per operator
and exposes :meth:`load_kwargs` — used by the Pipeline to build the
``operator.load(**kwargs)`` arguments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ShotScaleConfig:
    """Shot Scale — DINOv2 ViT-B/14."""
    ckpt: str = "shot_scale/dinov2_shot_scale.pth"
    backbone_ckpt: str = "shot_scale/dinov2_vitb14_pretrain.pth"


@dataclass
class CompositionConfig:
    """Composition — DINOv3 ViT-B/16."""
    ckpt: str = "composition/dinov3_composition.pth"
    backbone_ckpt: str = "composition/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


@dataclass
class CameraAngleConfig:
    """Camera Angle — BEiT-based multi-label classifier.

    Predictor code is bundled in the package; only trained weight files
    need to be placed under ``ckpt_subdir`` (relative to
    ``checkpoint_dir``). See README for the recommended layout.
    """
    # Subdirectory that contains ``shot_angle/`` and ``dutch_shot/`` weight
    # folders, relative to ``checkpoint_dir``.
    ckpt_subdir: str = "camera_angle"


@dataclass
class ColorToneConfig:
    """Color & Tone — ResNet-18 multilabel."""
    ckpt: str = "color_tone/resnet18_color_tone.pth"


@dataclass
class CharacterLayoutConfig:
    """Character Layout — InternVL3-14B + LoRA merged."""
    ckpt: str = "character_layout/character_layout_ckpt"


@dataclass
class CameraMovementConfig:
    """Camera Movement — InternVL3-14B + LoRA merged."""
    ckpt: str = "camera_movement/camera_movement_ckpt"


@dataclass
class OperatorConfigs:
    """Container for all per-operator configs."""

    shot_scale: ShotScaleConfig = field(default_factory=ShotScaleConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    camera_angle: CameraAngleConfig = field(default_factory=CameraAngleConfig)
    color_tone: ColorToneConfig = field(default_factory=ColorToneConfig)
    character_layout: CharacterLayoutConfig = field(default_factory=CharacterLayoutConfig)
    camera_movement: CameraMovementConfig = field(default_factory=CameraMovementConfig)

    def load_kwargs(
        self,
        name: str,
        base_checkpoint_dir: str,
        device: str,
    ) -> Dict[str, Any]:
        """Build the ``operator.load(**kwargs)`` arguments for ``name``."""
        ckpt_root = Path(base_checkpoint_dir)

        def _abs(rel: str) -> str:
            return str(ckpt_root / rel)

        if name == "shot_scale":
            c = self.shot_scale
            return {
                "model_path": _abs(c.ckpt),
                "dinov2_pretrained_path": _abs(c.backbone_ckpt),
                "device": device,
            }
        if name == "composition":
            c = self.composition
            return {
                "model_path": _abs(c.ckpt),
                "dinov3_pretrained_path": _abs(c.backbone_ckpt),
                "device": device,
            }
        if name == "camera_angle":
            c = self.camera_angle
            return {
                "ckpt_dir": _abs(c.ckpt_subdir),
                "device": device,
            }
        if name == "color_tone":
            c = self.color_tone
            return {"model_path": _abs(c.ckpt), "device": device}
        if name == "character_layout":
            c = self.character_layout
            return {"model_path": _abs(c.ckpt), "device": device}
        if name == "camera_movement":
            c = self.camera_movement
            return {"model_path": _abs(c.ckpt), "device": device}

        raise KeyError(f"No load_kwargs builder for operator {name!r}")
