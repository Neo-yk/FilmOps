"""Parse free-form MLLM responses into the camera-movement label set."""

from typing import List

from filmops.operators.camera_movement.prompts import MOVEMENT_LABELS, MOVEMENT_LABELS_EN


def parse_movement_response(text: str) -> List[str]:
    """Extract camera-movement labels from a model response string.

    The model responds in Chinese; matched Chinese labels are mapped to their
    user-facing English names via ``MOVEMENT_LABELS_EN``.
    """
    labels = [MOVEMENT_LABELS_EN[lbl] for lbl in MOVEMENT_LABELS if lbl in text]
    return labels or ["Unknown"]
