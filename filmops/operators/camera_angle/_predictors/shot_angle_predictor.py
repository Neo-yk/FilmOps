"""Camera-angle & aerial (bird's-eye) predictor — single-frame only.

Simplified from ``shot_orientation_classification/models/shot_angle/predictorv3.py``.
Only ``predict_file`` / ``predict_image`` are retained; all video-level
aggregation, batch processing and result-saving code has been removed.
"""

import logging
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from filmops.backbones.beit import BeitForImageClassification


class CameraAngleAerialPredictor:
    """Vertical angle + bird's-eye view predictor.

    Classes (index → label)::

        0: 鸟瞰/高空视角   (aerial / overhead)
        1: 极仰拍/垂直视角 (extreme low / below)
        2: 俯视            (high angle)
        3: 仰视            (low angle)
        4: 平视            (neutral / eye-level)
        5: 极俯拍/上帝视角 (extreme high / overhead)
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = (384, 384)
        self.class_list = [
            "鸟瞰/高空视角", "极仰拍/垂直视角", "俯视",
            "仰视", "平视", "极俯拍/上帝视角",
        ]
        self.label2id = {label: str(i) for i, label in enumerate(self.class_list)}
        self.id2label = {str(i): label for i, label in enumerate(self.class_list)}
        self.model = self._load_model(self.model_path)
        self.logger = logger or self._setup_default_logger()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        logger = logging.getLogger("CameraAngleAerialPredictor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_model(self, model_path: str) -> nn.Module:
        from transformers import BeitConfig

        model_path = os.path.abspath(model_path)
        config = BeitConfig.from_pretrained(model_path, local_files_only=True)
        model = BeitForImageClassification(config)
        # Explicitly load state_dict (from_pretrained may silently skip with
        # newer transformers versions).
        _bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(_bin_path):
            _state = torch.load(_bin_path, map_location="cpu")
            _model_keys = set(model.state_dict().keys())
            _filtered = {k: v for k, v in _state.items() if k in _model_keys}
            model.load_state_dict(_filtered, strict=False)

        self.transform = Compose([
            Resize(self.img_size),
            CenterCrop(self.img_size),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        model.to(self.device)
        model.eval()
        return model

    def get_max(self, sample_pred):
        """Return (label, confidence) for the best non-aerial class."""
        return (
            self.class_list[1:][np.argmax(sample_pred[1:])],
            np.max(sample_pred[1:]).item(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_file(
        self, image_path: str, return_confidence: bool = True
    ) -> Union[
        Tuple[Tuple[str, float], Tuple[str, float]],
        Tuple[str, str],
    ]:
        """Run inference on a single image file.

        Returns:
            ``(angle_result, aerial_result)`` where each is
            ``(label, confidence)`` when *return_confidence* is ``True``.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.predict_image(image, return_confidence)
        except Exception as e:
            self.logger.error("predict_file failed for %s: %s", image_path, e)
            if return_confidence:
                return ("预测失败", 0.0), ("预测失败", 0.0)
            return "预测失败", "预测失败"

    def predict_image(
        self, image: Union[Image.Image, np.ndarray], return_confidence: bool = True
    ) -> Union[
        Tuple[Tuple[str, float], Tuple[str, float]],
        Tuple[str, str],
    ]:
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor)
            probabilities = torch.sigmoid(outputs.logits)
            angle_predicted = [
                self.get_max(p) for p in probabilities.cpu().numpy().tolist()
            ]
            aerial_predicted = [
                ("鸟瞰/高空视角", p[0]) if p[0] > 0.5 else ("非鸟瞰", p[0])
                for p in probabilities.cpu().numpy().tolist()
            ]

        if return_confidence:
            return angle_predicted[0], aerial_predicted[0]
        return angle_predicted[0][0], aerial_predicted[0][0]
