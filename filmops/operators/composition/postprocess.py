"""Per-frame post-processing for the Composition operator."""

from typing import List

from filmops.operators.composition.model import LABEL_LIST, LABEL_THRESHOLDS


def post_process_frame(probs: List[float]) -> List[str]:
    """Apply per-class thresholds to a single frame's sigmoid outputs."""
    preds = [
        label for i, label in enumerate(LABEL_LIST)
        if probs[i] >= LABEL_THRESHOLDS.get(label, 0.5)
    ]
    return preds or ["Unknown"]
