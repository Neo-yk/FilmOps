"""Bundled predictor classes for the camera_angle operator.

These are simplified, frame-only versions of the predictors originally
from the ``shot_orientation_classification`` repository.  Only single-image
inference (``predict_file`` / ``predict_image``) is retained; all video
processing, batch handling and result-saving logic has been removed.
"""

from .shot_angle_predictor import CameraAngleAerialPredictor
from .dutch_shot_predictor import DutchShotPredictor

__all__ = ["CameraAngleAerialPredictor", "DutchShotPredictor"]
