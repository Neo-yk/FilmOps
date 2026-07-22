"""Dutch-angle (tilted camera) predictor — single-frame only.

Simplified from ``shot_orientation_classification/models/dutch_shot/predictorv2.py``.
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
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomResizedCrop,
    ToTensor,
)

from filmops.backbones.beit import BeitForImageClassification


class DutchShotPredictor:
    """Binary Dutch-angle classifier.

    Classes::

        0: 荷兰角（倾斜镜头）  (Dutch angle)
        1: 正常角度             (normal angle)
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
        self.class_list = ["荷兰角（倾斜镜头）", "正常角度"]
        self.label2id = {label: str(i) for i, label in enumerate(self.class_list)}
        self.id2label = {str(i): label for i, label in enumerate(self.class_list)}
        self.model = self._load_model(self.model_path)
        self.logger = logger or self._setup_default_logger()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        logger = logging.getLogger("DutchShotPredictor")
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

        # Dutch-angle model uses RandomResizedCrop (no aspect-ratio change)
        # rather than Resize+CenterCrop, matching the original training setup.
        self.transform = Compose([
            RandomResizedCrop(self.img_size, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_file(
        self, image_path: str, return_confidence: bool = True
    ) -> Union[str, Tuple[str, float]]:
        """Run inference on a single image file.

        Returns:
            ``(label, confidence)`` when *return_confidence* is ``True``.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.predict_image(image, return_confidence)
        except Exception as e:
            self.logger.error("predict_file failed for %s: %s", image_path, e)
            if return_confidence:
                return "预测失败", 0.0
            return "预测失败"

    def predict_image(
        self, image: Union[Image.Image, np.ndarray], return_confidence: bool = True
    ) -> Union[str, Tuple[str, float]]:
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.class_list[predicted.item()]
        confidence_score = confidence.item()
        if return_confidence:
            return predicted_class, confidence_score
        return predicted_class
