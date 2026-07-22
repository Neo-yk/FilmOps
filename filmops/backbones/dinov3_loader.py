"""DINOv3 ViT-B/16 backbone loader.

The DINOv3 model code is bundled in ``filmops.backbones.dinov3``.
Only the trained weight file needs to be supplied at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

# DINOv3 source uses absolute imports (``from dinov3.utils import …``).
# Ensure the bundled package directory is on *sys.path*.
_backbones_dir = str(Path(__file__).resolve().parent)
if _backbones_dir not in sys.path:
    sys.path.insert(0, _backbones_dir)

import torch
import torch.nn as nn

from dinov3.hub.backbones import dinov3_vitb16


def load_dinov3_vitb16(pretrained_path: str) -> nn.Module:
    """Load DINOv3 ViT-B/16 with pretrained backbone weights.

    Args:
        pretrained_path: Path to the DINOv3 pretrained weight file.
    """
    model = dinov3_vitb16(pretrained=False)
    state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model
