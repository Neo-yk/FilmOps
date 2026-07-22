"""Model definition and labels for the Composition operator."""

from typing import Dict, List

import torch
import torch.nn as nn

from filmops.backbones.dinov3_loader import load_dinov3_vitb16

# 12 composition labels + 1 explicit "NAN" (Unknown).
LABEL_LIST: List[str] = [
    "Center", "Rule of Thirds", "Horizontal", "Vertical", "Symmetric", "Framing",
    "Scattered", "Leading Lines", "Diagonal", "Oblique", "Triangular", "Depth of Field", "Unknown",
]
NUM_CLASSES = len(LABEL_LIST)

# Per-class optimal thresholds (from training-time calibration).
LABEL_THRESHOLDS: Dict[str, float] = {
    "Center": 0.132, "Rule of Thirds": 0.167, "Horizontal": 0.242, "Vertical": 0.508,
    "Symmetric": 0.230, "Framing": 0.383, "Scattered": 0.209, "Leading Lines": 0.284,
    "Diagonal": 0.243, "Oblique": 0.193, "Triangular": 0.710, "Depth of Field": 0.310,
    "Unknown": 0.211,
}


class Dinov3Classifier(nn.Module):
    """DINOv3 ViT-B/16 with a 3-layer GELU/BN MLP classification head."""

    def __init__(self, pretrained_path: str, num_classes: int):
        super().__init__()
        self.transformer = load_dinov3_vitb16(pretrained_path)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 512), nn.GELU(), nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.transformer(x))
