"""Prompt templates for the Character Layout operator."""

# Single-image mode: keyframe only.
PROMPT_SINGLE = (
    "<image>\n"
    "请描述图片中每个人物的画面站位和朝向。\n"
    "位置以画面中的相对区域表示（如画面中心、左三分线附近、右下角等）；"
    "朝向以人物面部相对画面的方向表示（如正面朝向画面前方、侧面朝左等）。"
)

# Multi-image mode: keyframe + per-character reference crops.
PROMPT_MULTI = (
    "给你数张图片，图1为关键帧图像，图2为character[0]角色参考图，"
    "图3为character[1]角色参考图，依此类推\n"
    "请给出每个角色在关键帧图像中的朝向、站位和z轴站位（由近到远）\n"
    "当只有一个角色时，z轴站位输出无，"
    "当无法判断某角色图在关键帧中的信息时，输出无"
)
