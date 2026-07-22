"""Composition — DINOv3 ViT-B/16, 12-class multi-label classifier."""

from typing import Any, Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms

from filmops.core.base import BaseOperator
from filmops.core.registry import register_operator
from filmops.core.types import INPUT_FRAMES
from filmops.operators.composition.model import NUM_CLASSES, Dinov3Classifier
from filmops.operators.composition.postprocess import post_process_frame


@register_operator("composition")
class CompositionOperator(BaseOperator):
    """Composition classifier.

    Backbone: DINOv3 ViT-B/16, fine-tuned for 12-class multi-label classification.
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
        dinov3_pretrained_path: str,
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

        self.model = Dinov3Classifier(dinov3_pretrained_path, NUM_CLASSES)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device).eval()

    def _infer_batch(self, frames: List[np.ndarray], batch_size: int = 8) -> List[List[float]]:
        all_probs: List[List[float]] = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            tensor_batch = torch.stack([self.data_tsf(f) for f in batch]).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                if probs.ndim == 1:
                    probs = np.expand_dims(probs, axis=0)
                all_probs.extend(probs.tolist())
        return all_probs

    def predict(self, inputs: Any, **kwargs) -> Dict[str, Any]:
        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:
            inputs = [inputs]

        probs_list = self._infer_batch(inputs)
        labels = post_process_frame(probs_list[0])

        return {"labels": labels, "raw_probs": probs_list}
