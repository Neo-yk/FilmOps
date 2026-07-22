"""Common type aliases and structured input/output containers."""

from typing import Any, Dict, List, Union

import numpy as np

# An operator may receive any of the following at predict() time:
#   - a single np.ndarray (H, W, 3 RGB)
#   - a list of np.ndarray frames
#   - a single file path (image or video)
#   - a list of file paths
#   - a dict with operator-specific fields (e.g. {"frame": ..., "characters": [...]})
OperatorInput = Union[np.ndarray, List[np.ndarray], str, List[str], Dict[str, Any]]


# Operator input mode declarations. Used by the Pipeline to dispatch inputs
# without hard-coded if-elif over operator names.
INPUT_FRAMES = "frames"              # list of np.ndarray frames
INPUT_FRAME_PATHS = "frame_paths"    # list of str paths
INPUT_VIDEO_PATH = "video_path"      # single str path to video
INPUT_CUSTOM = "custom"              # caller passes raw input dict
