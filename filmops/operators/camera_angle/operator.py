"""Camera Angle — BEiT-based, 7-class multi-label classifier.

This operator wraps two bundled sub-models:

* ``CameraAngleAerialPredictor`` — vertical angle (5 classes) + bird's-eye (binary)
* ``DutchShotPredictor``         — Dutch angle (binary)

The predictor code is bundled under ``_predictors/``; only the trained
weight files need to be placed in ``checkpoints/camera_angle/``.
"""

import os
from typing import Any, Dict, List

from filmops.core.base import BaseOperator
from filmops.core.exceptions import OperatorLoadError
from filmops.core.registry import register_operator
from filmops.core.types import INPUT_FRAME_PATHS
from filmops.operators.camera_angle._predictors import (
    CameraAngleAerialPredictor,
    DutchShotPredictor,
)
from filmops.operators.camera_angle.thresholds import (
    AERIAL_THRESHOLDS,
    ANGLE_MAP,
    DUTCH_THRESHOLDS,
    SHOT_ANGLE_THRESHOLDS,
)


@register_operator("camera_angle")
class CameraAngleOperator(BaseOperator):
    """Camera angle classifier.

    Exposes the vertical-angle subset of the upstream model
    (Eye Level / Low Angle / High Angle / Bird's Eye / Extreme Low Angle /
    Extreme High Angle / Dutch Angle).
    """

    granularity = "frame"
    input_mode = INPUT_FRAME_PATHS  # expects image file paths

    def __init__(self):
        self.camera_predictor = None
        self.dutch_predictor = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        ckpt_dir: str,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        """Load the bundled predictor sub-models.

        Args:
            ckpt_dir: Path to the directory containing weight folders
                ``shot_angle/`` and ``dutch_shot/``.
            device: Target device (handled internally by the predictors).
        """
        if not os.path.isdir(ckpt_dir):
            raise OperatorLoadError(
                f"camera_angle: weights directory does not exist: {ckpt_dir!r}."
            )

        ckpt_dir = os.path.abspath(ckpt_dir)
        shot_angle_ckpt = os.path.join(ckpt_dir, "shot_angle")
        dutch_ckpt = os.path.join(ckpt_dir, "dutch_shot")

        try:
            self.camera_predictor = CameraAngleAerialPredictor(shot_angle_ckpt)
            self.dutch_predictor = DutchShotPredictor(dutch_ckpt)
        except Exception as e:
            raise OperatorLoadError(
                f"camera_angle: failed to load predictors from {ckpt_dir!r}. "
                f"Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Single-frame inference
    # ------------------------------------------------------------------

    def _predict_single(self, image_path: str) -> Dict[str, Any]:
        cam_result, aerial_result = self.camera_predictor.predict_file(
            image_path, return_confidence=True,
        )
        cam_class, cam_conf = (
            cam_result if cam_result[0] != "预测失败" else ("预测失败", 0.0)
        )
        aerial_class, aerial_conf = (
            aerial_result if aerial_result[0] != "预测失败" else ("预测失败", 0.0)
        )

        try:
            dutch_class, dutch_conf = self.dutch_predictor.predict_file(
                image_path, return_confidence=True,
            )
        except Exception:
            dutch_class, dutch_conf = "预测失败", 0.0

        # Suppress conflicting predictions at extreme angles.
        if cam_class in ("极仰拍/垂直视角", "极俯拍/上帝视角"):
            dutch_class = "正常角度"
        if cam_class in ("极仰拍/垂直视角", "极俯拍/上帝视角", "仰视"):
            aerial_class = "非鸟瞰"

        return {
            "camera": (cam_class, cam_conf),
            "aerial": (aerial_class, aerial_conf),
            "dutch": (dutch_class, dutch_conf),
        }

    def _collect_labels(self, raw: Dict[str, Any]) -> List[str]:
        labels: List[str] = []
        for key, thresholds in (
            ("camera", SHOT_ANGLE_THRESHOLDS),
            ("aerial", AERIAL_THRESHOLDS),
            ("dutch", DUTCH_THRESHOLDS),
        ):
            cls, conf = raw[key]
            if conf >= thresholds.get(cls, 1.0):
                mapped = ANGLE_MAP.get(cls)
                if mapped:
                    labels.append(mapped)
        return labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, inputs: Any, **kwargs) -> Dict[str, Any]:
        # Accept a single image path (str or Path).
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]

        raw = self._predict_single(str(inputs))
        labels = self._collect_labels(raw) or ["Unknown"]

        return {"labels": labels, "raw": raw}
