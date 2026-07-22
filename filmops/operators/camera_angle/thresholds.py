"""Label mapping and per-class thresholds for the Camera Angle operator."""

from typing import Dict

# External predictor's labels (Chinese, emitted by the upstream
# shot_orientation_classification model) -> FilmOps taxonomy (English).
# The KEYS must match the external model's raw output verbatim and must
# stay in Chinese; only the mapped VALUES are user-facing.
ANGLE_MAP: Dict[str, str] = {
    "平视": "Eye Level",
    "俯视": "High Angle",
    "仰视": "Low Angle",
    "极俯拍/上帝视角": "Extreme High Angle",
    "极仰拍/垂直视角": "Extreme Low Angle",
    "鸟瞰/高空视角": "Bird's Eye",
    "荷兰角（倾斜镜头）": "Dutch Angle",
}

# Per-class confidence thresholds (0.5).
SHOT_ANGLE_THRESHOLDS: Dict[str, float] = {
    "极俯拍/上帝视角": 0.5, "俯视": 0.5, "平视": 0.5,
    "仰视": 0.5, "极仰拍/垂直视角": 0.5,
}
AERIAL_THRESHOLDS: Dict[str, float] = {"鸟瞰/高空视角": 0.5, "非鸟瞰": 0.0}
DUTCH_THRESHOLDS: Dict[str, float] = {"荷兰角（倾斜镜头）": 0.5, "正常角度": 0.0}
