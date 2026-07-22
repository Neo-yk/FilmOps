"""Model definition and label mapping for the Shot Scale operator."""

import torch.nn as nn

# 8 shot scales + 1 explicit "NAN" (Unknown) class.
NUM_CLASSES = 9

# Class index -> abbreviation / full label. The classifier head emits the
# index; ``LABEL_MAP`` is the user-facing name used everywhere downstream.
LABEL_ABBR = {
    0: "CS", 1: "CU", 2: "ECU", 3: "ELS",
    4: "FS", 5: "LS", 6: "MFS", 7: "MS", 8: "NAN",
}
LABEL_MAP = {
    0: "Close Shot", 1: "Close-Up", 2: "Extreme Close-Up", 3: "Extreme Long Shot",
    4: "Full Shot", 5: "Long Shot", 6: "Medium Full Shot", 7: "Medium Shot", 8: "Unknown",
}


class Dinov2Classifier(nn.Module):
    """DINOv2 ViT-B/14 backbone with a 2-layer MLP classification head."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.transformer = base_model
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        return self.classifier(x)
