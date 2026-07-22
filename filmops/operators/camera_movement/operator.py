"""Camera Movement — InternVL3-14B + LoRA SFT.

Shot-level operator that classifies camera motion into 10 categories:
Push In / Pull Out / Pan / Tracking / Static / Arc / Follow / Roll / Zoom / Dynamic.
"""

from typing import Any, Dict, List

import torch

from filmops.backbones.internvl_loader import load_internvl_model
from filmops.core.base import BaseOperator
from filmops.core.registry import register_operator
from filmops.core.types import INPUT_VIDEO_PATH
from filmops.operators.camera_movement.parser import parse_movement_response
from filmops.operators.camera_movement.prompts import PROMPT
from filmops.preprocessing.internvl_tiles import load_video_tiles


@register_operator("camera_movement")
class CameraMovementOperator(BaseOperator):
    """Camera movement classifier (shot-level)."""

    granularity = "shot"
    input_mode = INPUT_VIDEO_PATH

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self._device = None

    # ------------------------------------------------------------------

    def load(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        self._device = device
        self.model, self.tokenizer = load_internvl_model(model_path, device=device)
        self.generation_config = dict(
            max_new_tokens=512,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # ------------------------------------------------------------------

    def predict(self, inputs: Any, **kwargs) -> Dict[str, Any]:
        """Predict camera movement labels for a video shot (or list of shots)."""
        if isinstance(inputs, list):
            per_shot = [self._predict_single(p) for p in inputs]
            all_labels: List[str] = []
            for r in per_shot:
                for lbl in r["labels"]:
                    if lbl not in all_labels:
                        all_labels.append(lbl)
            return {"labels": all_labels, "per_shot": per_shot}
        return self._predict_single(inputs)

    # ------------------------------------------------------------------

    def _predict_single(self, video_path: str) -> Dict[str, Any]:
        pixel_values, num_patches_list = load_video_tiles(
            video_path, num_segments=16, max_num=1,
        )
        pixel_values = pixel_values.to(torch.bfloat16).to(self._device)

        video_prefix = "".join(
            f"Frame-{i+1}: <image>\n" for i in range(len(num_patches_list))
        )
        question = video_prefix + PROMPT

        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                self.generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )

        raw_text = response[0].strip() if isinstance(response, tuple) else str(response).strip()
        return {"labels": parse_movement_response(raw_text), "raw": raw_text}
