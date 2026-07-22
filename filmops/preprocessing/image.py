"""Image-level preprocessing helpers."""

from __future__ import annotations

import cv2
import numpy as np


def load_image(image_path: str, short_side: int = 256) -> np.ndarray:
    """Load an image file as an RGB numpy array, optionally downscaled.

    Args:
        image_path: Path to the image file.
        short_side: If the smaller image dimension exceeds this value, the
            image is downscaled so its shorter side equals ``short_side``.
            Pass ``0`` to skip resizing.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if short_side > 0:
        h, w = img.shape[:2]
        if min(w, h) > short_side:
            scale = short_side / min(w, h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img
