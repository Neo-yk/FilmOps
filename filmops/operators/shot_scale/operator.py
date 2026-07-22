"""Shot Scale — DINOv2 ViT-B/14, 8-class single-label classifier."""

from typing import Any, Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms

from filmops.backbones.dinov2_loader import load_dinov2_vitb14
from filmops.core.base import BaseOperator
from filmops.core.registry import register_operator
from filmops.core.types import INPUT_FRAMES
from filmops.operators.shot_scale.model import (
    LABEL_MAP,
    Dinov2Classifier,
)


@register_operator("shot_scale")
class ShotScaleOperator(BaseOperator):
    """Shot scale classifier.

    Backbone: DINOv2 ViT-B/14, fine-tuned for 8-class single-label classification.
    Input: single frame or list of frames.
    Output: one of Extreme Close-Up / Close-Up / Close Shot / Medium Shot /
    Medium Full Shot / Full Shot / Long Shot / Extreme Long Shot.
    """

    granularity = "frame"
    input_mode = INPUT_FRAMES

    def __init__(self):
        self.model = None
        self.device = None
        self.data_tsf = None

    def load(
        self,
        model_path: str,
        dinov2_pretrained_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        IMAGE_SIZE = (518, 518)
        self.data_tsf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        base_model = load_dinov2_vitb14(dinov2_pretrained_path, self.device)
        self.model = Dinov2Classifier(base_model)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device).eval()

    def _infer_batch(self, frames: List[np.ndarray], batch_size: int = 8) -> List[int]:
        """Run batched inference; return per-frame class indices with calibration."""
        all_preds: List[int] = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            tensor_batch = torch.stack([self.data_tsf(f) for f in batch])
            with torch.no_grad():
                logits = self.model(tensor_batch.to(self.device))
                probs = torch.nn.functional.softmax(logits, dim=1) * 100
                _, preds = torch.max(logits, 1)
                preds = preds.cpu().tolist()
                probs = probs.cpu().tolist()

            # Confidence-based remapping for ambiguous boundary classes.
            thresholds = [20, 20, 15, 90]
            for pred, prob in zip(preds, probs):
                if pred == 4 and prob[6] > thresholds[0]:    # FS → MFS
                    all_preds.append(6)
                elif pred == 5 and prob[4] > thresholds[1]:  # LS → FS
                    all_preds.append(4)
                elif pred == 2 and prob[1] > thresholds[2]:  # ECU → CU
                    all_preds.append(1)
                elif pred == 3 and prob[3] < thresholds[3]:  # ELS → NAN
                    all_preds.append(8)
                else:
                    all_preds.append(pred)
        return all_preds

    def predict(self, inputs: Any, **kwargs) -> Dict[str, Any]:
        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:
            inputs = [inputs]

        preds = self._infer_batch(inputs)
        labels = [LABEL_MAP.get(preds[0], "Unknown")]

        return {"labels": labels, "raw_indices": preds}
