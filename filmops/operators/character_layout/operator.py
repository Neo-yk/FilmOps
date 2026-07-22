"""Character Layout — InternVL3-14B + LoRA SFT.

Produces natural-language descriptions of character positions and
orientations in the frame coordinate system. Supports two input modes:

* single-image mode: a frame path only
* multi-image mode: a frame + N character reference crops
"""

from typing import Any, Dict, List

import torch

from filmops.backbones.internvl_loader import load_internvl_model
from filmops.core.base import BaseOperator
from filmops.core.registry import register_operator
from filmops.core.types import INPUT_CUSTOM
from filmops.operators.character_layout.prompts import PROMPT_MULTI, PROMPT_SINGLE
from filmops.preprocessing.internvl_tiles import load_image_tiles


@register_operator("character_layout")
class CharacterLayoutOperator(BaseOperator):
    """Character layout captioner."""

    granularity = "frame"
    input_mode = INPUT_CUSTOM  # accepts str / dict / list of either

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self._device = None

    # ------------------------------------------------------------------

    def load(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        self._device = device
        self.model, self.tokenizer = load_internvl_model(model_path, device=device)
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)

    # ------------------------------------------------------------------

    def predict(self, inputs: Any, **kwargs) -> Dict[str, Any]:
        """Predict character layout description(s).

        Accepts any of:
            - str:  single frame image path (single-image mode).
            - dict: ``{"frame": str, "characters": [str, ...]}`` (multi-image mode).
            - list of str: multiple frame paths (single-image mode for each).
            - list of dict: multiple multi-image inputs.
        """
        items: List[Any] = [inputs] if isinstance(inputs, (str, dict)) else list(inputs)

        descriptions: List[str] = []
        raw_responses: List[str] = []

        for item in items:
            if isinstance(item, dict):
                resp = self._predict_multi(item["frame"], item.get("characters", []))
            else:
                resp = self._predict_single(str(item))
            descriptions.append(resp.strip())
            raw_responses.append(resp)

        return {"labels": descriptions, "raw": raw_responses}

    # ------------------------------------------------------------------

    def _predict_single(self, image_path: str) -> str:
        pixel_values = (
            load_image_tiles(image_path, max_num=12)
            .to(torch.bfloat16)
            .to(self._device)
        )
        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                PROMPT_SINGLE,
                self.generation_config,
            )
        if isinstance(response, tuple):
            return response[0].strip()
        return response.strip()

    def _predict_multi(self, frame_path: str, character_paths: List[str]) -> str:
        image_num = 1 + len(character_paths)
        all_paths = [frame_path, *character_paths]
        max_num = max(1, 12 // image_num)

        pixel_values_list = []
        num_patches_list = []
        for path in all_paths:
            pv = (
                load_image_tiles(path, max_num=max_num)
                .to(torch.bfloat16)
                .to(self._device)
            )
            pixel_values_list.append(pv)
            num_patches_list.append(pv.size(0))

        pixel_values = torch.cat(pixel_values_list, dim=0)
        question = "<image>\n" * image_num + PROMPT_MULTI

        with torch.inference_mode():
            result = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                self.generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
        if isinstance(result, tuple):
            return result[0].strip()
        return result.strip()
