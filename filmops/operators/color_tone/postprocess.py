"""Per-frame post-processing for the Color & Tone operator."""

from typing import List, Tuple

from filmops.operators.color_tone.model import (
    HUE_NEIGHBOURS,
    LABEL_LIST,
    LABEL_THRESHOLDS,
)


def _max_with_knn(values: List[float], knn_ratio: float = 1.5) -> Tuple[int, int, bool]:
    """Return ``(idx_max, idx_second, max/second > knn_ratio)``."""
    sorted_vals = sorted(values)
    mx, sec = sorted_vals[-1], sorted_vals[-2]
    idx_mx = values.index(mx)
    idx_sec = values.index(sec)
    knn = mx / (sec + 1e-10) > knn_ratio
    return idx_mx, idx_sec, knn


def post_process_frame(output: List[float]) -> List[str]:
    """Map raw sigmoid outputs to a multi-label string list."""
    pred: List[str] = []

    # Tone (indices 0-2)
    tone_pred = output[0:3]
    tone_idx, _, knn_tone = _max_with_knn(tone_pred, knn_ratio=1.5)
    pred.append(LABEL_LIST[tone_idx] if knn_tone else "Mixed")

    # Saturation (indices 3-4)
    sat_pred = output[3:5]
    sat_idx, _, knn_sat = _max_with_knn(sat_pred, knn_ratio=1.5)
    if knn_sat:
        pred.append(LABEL_LIST[sat_idx + 3])

    # Hue (indices 5..)
    color_pred = output[5:]
    c_max_idx, c_sec_idx, _ = _max_with_knn(color_pred)
    c_max_val = color_pred[c_max_idx]
    c_sec_val = color_pred[c_sec_idx]
    c_max_lbl = LABEL_LIST[c_max_idx + 5]
    c_sec_lbl = LABEL_LIST[c_sec_idx + 5]

    cond1 = c_max_val > LABEL_THRESHOLDS.get(c_max_lbl, 0.5)
    cond2 = (
        c_sec_lbl in HUE_NEIGHBOURS.get(c_max_lbl, [])
        and c_max_val > LABEL_THRESHOLDS.get(c_max_lbl, 0.5) * 0.7
    )
    if c_max_lbl == "Pink":
        cond2 = False

    if cond1 or cond2:
        pred.append(c_max_lbl)
        knn_color = c_max_val / (c_sec_val + 1e-10)
        threshold_ratio = (
            1.25 * LABEL_THRESHOLDS.get(c_max_lbl, 0.5)
            / LABEL_THRESHOLDS.get(c_sec_lbl, 0.5)
        )
        if knn_color < threshold_ratio and c_sec_lbl != "Pink":
            pred.append(c_sec_lbl)

    # "Black" is a raw-head class that is intentionally not exposed as a
    # user-facing label (see model.ORDERED_OUTPUT); drop it like "NONE".
    return [p for p in pred if p not in ("NONE", "Black")]
