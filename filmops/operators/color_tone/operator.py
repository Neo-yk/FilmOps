"""Color & Tone — ResNet-18 multi-label classifier (18 labels).

Three orthogonal sub-axes: hue (12), color temperature (3), saturation (3).
The underlying network has a 19-way head, but two raw classes ("Black" and
"NONE") are filtered out, leaving 18 user-facing labels.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms

from filmops.core.base import BaseOperator
from filmops.core.registry import register_operator
from filmops.core.types import INPUT_FRAMES
from filmops.operators.color_tone.model import NUM_CLASSES, ResNet18MultiLabel
from filmops.operators.color_tone.postprocess import post_process_frame


@register_operator("color_tone")
class ColorToneOperator(BaseOperator):
    """Color & Tone classifier."""

    granularity = "frame"
    input_mode = INPUT_FRAMES

    def __init__(self):
        self.model = None
        self.device = None
        self.data_tsf = None

    def load(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = ResNet18MultiLabel(NUM_CLASSES)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device).eval()

        self.data_tsf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

    def _infer_batch(self, frames: List[np.ndarray], batch_size: int = 8) -> List[List[float]]:
        all_outputs: List[List[float]] = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            tensor_batch = torch.stack([self.data_tsf(f) for f in batch])
            with torch.no_grad():
                out = self.model(tensor_batch.to(self.device))
                all_outputs.extend(out.cpu().tolist())
        return all_outputs

    def predict(self, inputs: Any, **kwargs) -> Dict[str, Any]:
        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:
            inputs = [inputs]

        outputs = self._infer_batch(inputs)
        labels = post_process_frame(outputs[0])

        return {"labels": labels, "raw_probs": outputs}
