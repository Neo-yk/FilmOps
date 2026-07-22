"""DINOv2 ViT-B/14 backbone loader.

The DINOv2 model code is bundled in ``filmops.backbones.dinov2``.
Only the trained weight file (``dinov2_vitb14_pretrain.pth``) needs to
be supplied at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

# DINOv2 source uses absolute imports (``from dinov2.layers import …``).
# Ensure the bundled package directory is on *sys.path*.
_backbones_dir = str(Path(__file__).resolve().parent)
if _backbones_dir not in sys.path:
    sys.path.insert(0, _backbones_dir)

import torch
import torch.nn as nn

from dinov2.hub.backbones import dinov2_vitb14


def load_dinov2_vitb14(
    pretrained_path: str,
    device: torch.device,
) -> nn.Module:
    """Build a DINOv2 ViT-B/14 and load pretrained weights.

    Args:
        pretrained_path: Path to ``dinov2_vitb14_pretrain.pth``.
        device: Target torch device.
    """
    model = dinov2_vitb14(pretrained=False)
    state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)
