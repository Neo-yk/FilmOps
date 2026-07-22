"""Model definition, labels, and thresholds for the Color & Tone operator."""

from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models import resnet18

# Label indices:
#   0..2  : color temperature (Warm / Cool / Mixed)
#   3..4  : saturation (High Saturation / Low Saturation)
#   5..18 : hue (12 colors + Monochrome / NONE)
LABEL_LIST: List[str] = [
    "Warm", "Cool", "Mixed",
    "High Saturation", "Low Saturation",
    "Red", "Orange", "Yellow", "Green", "Cyan", "Blue",
    "Purple", "Magenta", "Pink", "Brown", "Black", "White",
    "Monochrome", "NONE",
]
NUM_CLASSES = len(LABEL_LIST)  # 19

# Per-class thresholds (calibrated).
LABEL_THRESHOLDS: Dict[str, float] = {
    "Warm": 0.3, "Cool": 0.3, "Mixed": 0.3,
    "High Saturation": 0.2, "Low Saturation": 0.3,
    "Red": 0.4, "Orange": 0.33, "Yellow": 0.4, "Green": 0.36,
    "Cyan": 0.42, "Blue": 0.42, "Purple": 0.6, "Magenta": 0.36,
    "Pink": 0.85, "Brown": 0.4, "Black": 0.75, "White": 0.85,
    "Monochrome": 0.85, "NONE": 0.3,
}

# Hue neighbour relationships (for color-smoothing rules).
HUE_NEIGHBOURS: Dict[str, List[str]] = {
    "Red": ["Orange", "Magenta"], "Orange": ["Red", "Yellow", "Brown"],
    "Yellow": ["Orange", "Green", "Brown"], "Green": ["Cyan", "Yellow"],
    "Cyan": ["Green", "Blue"], "Blue": ["Cyan", "Purple"],
    "Purple": ["Red", "Blue", "Magenta", "Pink"], "Magenta": ["Red", "Magenta", "Pink"],
    "Pink": ["Purple", "Magenta"], "Brown": ["Red", "Yellow", "Orange"],
    "Black": [], "White": [], "Monochrome": [], "NONE": [],
}

# Final output ordering for user-facing labels (18 total: 3 tone + 3
# saturation + 12 hue). The raw 19-way head also produces "Black" and
# "NONE", but both are filtered out of user-facing results (see
# ``postprocess.post_process_frame``), so they never appear here.
ORDERED_OUTPUT: List[str] = [
    "Warm", "Cool", "Mixed",
    "High Saturation", "Medium Saturation", "Low Saturation",
    "Red", "Orange", "Yellow", "Green", "Cyan", "Blue",
    "Purple", "Magenta", "Pink", "Brown", "White", "Monochrome",
]


class ResNet18MultiLabel(nn.Module):
    """ResNet-18 with a single ``num_classes``-way head, sigmoid-activated."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))
