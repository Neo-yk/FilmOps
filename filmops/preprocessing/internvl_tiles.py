"""InternVL dynamic-tile preprocessing.

Shared by :mod:`filmops.operators.character_layout` and
:mod:`filmops.operators.camera_movement`. The implementation follows the
official InternVL reference: build a normalized image transform, find the
best aspect-ratio tiling within ``[min_num, max_num]``, then crop / resize
accordingly. Optionally append a downsampled thumbnail.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transform(input_size: int = 448) -> T.Compose:
    """Build the standard InternVL image transform (RGB → normalized tensor)."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dynamic tiling
# ---------------------------------------------------------------------------

def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """Tile ``image`` into ``image_size`` squares using the best aspect ratio."""
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    best_ratio_diff = float("inf")
    best_ratio: Tuple[int, int] = (1, 1)
    area = orig_w * orig_h
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio

    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized = image.resize((target_w, target_h))

    tiles: List[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_w // image_size)) * image_size,
            (i // (target_w // image_size)) * image_size,
            ((i % (target_w // image_size)) + 1) * image_size,
            ((i // (target_w // image_size)) + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------

def load_image_tiles(
    image_file: str,
    input_size: int = 448,
    max_num: int = 12,
) -> torch.Tensor:
    """Load a single image file and return its tiled pixel-value tensor."""
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


def load_video_tiles(
    video_path: str,
    bound: Optional[Tuple[float, float]] = None,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Sample frames from a video and return InternVL-ready pixel values.

    Args:
        video_path: Path to the source video.
        bound: Optional ``(start_seconds, end_seconds)`` window.
        input_size: Tile side length in pixels.
        max_num: Maximum number of tiles per sampled frame.
        num_segments: Number of frames to sample. If ``None``, a duration-aware
            default in ``[16, 56]`` is computed (4 fps capped to the range).

    Returns:
        Tuple of ``(pixel_values, num_patches_list)``.
    """
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    if num_segments is None:
        seg = int((max_frame / fps) * 4) if fps > 0 else 16
        num_segments = min(max(seg, 16), 56)

    transform = build_transform(input_size)
    indices = _sample_indices(bound, fps, max_frame, num_segments=num_segments)

    pixel_values_list: List[torch.Tensor] = []
    num_patches_list: List[int] = []
    for fi in indices:
        img = Image.fromarray(vr[fi].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num,
        )
        pv = torch.stack([transform(t) for t in tiles])
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)

    return torch.cat(pixel_values_list), num_patches_list


def _sample_indices(
    bound: Optional[Tuple[float, float]],
    fps: float,
    max_frame: int,
    num_segments: int,
    first_idx: int = 0,
) -> np.ndarray:
    """Compute evenly-spaced frame indices within ``[bound]`` (or full range)."""
    if bound:
        start, end = bound
    else:
        start, end = -100000.0, 100000.0
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([
        int(start_idx + seg_size / 2 + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
