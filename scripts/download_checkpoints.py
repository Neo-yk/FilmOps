#!/usr/bin/env python3
"""Download FilmOps checkpoints.

Pretrained backbones (DINOv2 / DINOv3) are fetched from Meta's official CDN.
Fine-tuned weights are fetched from the FilmOps HuggingFace repo.

Usage::

    python scripts/download_checkpoints.py --dest ./checkpoints
"""

import argparse
import os
import sys
import urllib.request
from typing import Dict

# ── Official pretrained backbones (Meta) ──────────────────────────────
PRETRAINED_URLS: Dict[str, str] = {
    "shot_scale/dinov2_vitb14_pretrain.pth":
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "composition/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth":
        "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}

# ── Fine-tuned weights (FilmOps HF repo) ─────────────────────────────
# TODO: fill in your HuggingFace repo URL after publishing
HF_BASE = ""  # e.g. "https://huggingface.co/your-org/filmops-checkpoints/resolve/main"

FINETUNED_URLS: Dict[str, str] = {
    # "shot_scale/dinov2_shot_scale.pth": f"{HF_BASE}/shot_scale/dinov2_shot_scale.pth",
    # "composition/dinov3_composition.pth": f"{HF_BASE}/composition/dinov3_composition.pth",
    # "color_tone/resnet18_color_tone.pth": f"{HF_BASE}/color_tone/resnet18_color_tone.pth",
    # "camera_angle/shot_angle/pytorch_model.bin": ...,
    # "camera_angle/dutch_shot/pytorch_model.bin": ...,
    # "camera_angle/shot_angle/config.json": ...,
    # "camera_angle/dutch_shot/config.json": ...,
    # ... (InternVL merged checkpoints are large — recommend huggingface-cli)
}

ALL_URLS = {**PRETRAINED_URLS, **FINETUNED_URLS}


def main() -> int:
    p = argparse.ArgumentParser(description="Download FilmOps checkpoints.")
    p.add_argument("--dest", default="./checkpoints")
    p.add_argument("--pretrained-only", action="store_true",
                    help="Only download official pretrained backbones (skip fine-tuned weights)")
    args = p.parse_args()

    urls = PRETRAINED_URLS if args.pretrained_only else ALL_URLS

    if not urls:
        print("No URLs configured. Edit scripts/download_checkpoints.py to add them.")
        return 1

    os.makedirs(args.dest, exist_ok=True)
    for rel, url in urls.items():
        out = os.path.join(args.dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            print(f"[skip] {rel}")
            continue
        print(f"[get ] {rel}  ←  {url}")
        urllib.request.urlretrieve(url, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
