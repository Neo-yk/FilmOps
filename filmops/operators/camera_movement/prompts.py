"""Prompt template and label set for the Camera Movement operator.

The underlying InternVL3 model was fine-tuned to respond in Chinese, so both
the ``PROMPT`` and the ``MOVEMENT_LABELS`` matched against the raw response
MUST stay in Chinese. ``MOVEMENT_LABELS_EN`` maps each matched Chinese label
to its user-facing English name (applied by ``parser.parse_movement_response``).
"""

from typing import Dict, List

MOVEMENT_LABELS: List[str] = [
    "推", "拉", "摇", "移", "固定", "环绕", "跟随", "旋转", "变焦", "强运镜",
]

# Chinese (raw model output) -> English (user-facing) label mapping.
MOVEMENT_LABELS_EN: Dict[str, str] = {
    "推": "Push In", "拉": "Pull Out", "摇": "Pan", "移": "Tracking", "固定": "Static",
    "环绕": "Arc", "跟随": "Follow", "旋转": "Roll", "变焦": "Zoom", "强运镜": "Dynamic",
}

PROMPT = (
    "判断镜头运动方式属于下面哪种:摇, 移, 变焦, 跟随, 固定, 环绕, 推, 强运镜, 无法判断, 拉, 旋转中的哪几种"
)
