"""Unified FilmOps pipeline.

The Pipeline is intentionally agnostic of any specific operator. It walks
the registry, asks each registered operator class for its config, calls
``load(**config)``, and at predict time dispatches inputs according to
each operator's declared :attr:`input_mode`.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from filmops.config import FilmOpsConfig
from filmops.core.base import BaseOperator
from filmops.core.exceptions import OperatorInferenceError, OperatorLoadError
from filmops.core.registry import list_operators, resolve_operator
from filmops.core.types import INPUT_CUSTOM, INPUT_FRAME_PATHS, INPUT_FRAMES
from filmops.preprocessing.image import load_image

logger = logging.getLogger(__name__)


class FilmOpsPipeline:
    """Orchestrates all FilmOps operators for video/image analysis.

    Example::

        from filmops import FilmOpsConfig, FilmOpsPipeline

        cfg = FilmOpsConfig(checkpoint_dir="./checkpoints")
        pipe = FilmOpsPipeline(cfg)
        pipe.load()

        result = pipe.analyse_image("frame.jpg")
        result = pipe.analyse_video("shot.mp4")
    """

    def __init__(self, config: Optional[FilmOpsConfig] = None):
        self.config = config or FilmOpsConfig()
        self.operators: Dict[str, BaseOperator] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, names: Optional[List[str]] = None) -> None:
        """Load all enabled operators.

        Args:
            names: Optional explicit list to load. Defaults to
                ``self.config.enabled_operators`` (or all registered if None).
        """
        wanted = names or self.config.enabled_operators or list_operators()

        for name in wanted:
            if not self.config.is_operator_enabled(name):
                continue
            try:
                op_cls = resolve_operator(name)
                op = op_cls()
                load_kwargs = self.config.operator_configs.load_kwargs(
                    name, base_checkpoint_dir=self.config.checkpoint_dir,
                    device=self.config.device,
                )
                op.load(**load_kwargs)
                self.operators[name] = op
                logger.info("Loaded operator: %s", name)
            except Exception as e:
                logger.error("Failed to load operator %s: %s", name, e)
                raise OperatorLoadError(f"{name}: {e}") from e

        logger.info(
            "Pipeline ready with %d operators: %s",
            len(self.operators), list(self.operators.keys()),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def analyse_image(self, image_path: str) -> Dict[str, Any]:
        """Run all frame-level operators on a single image."""
        frame = load_image(image_path)
        results: Dict[str, Any] = {}

        for name, op in self.operators.items():
            if op.granularity != "frame":
                continue
            try:
                results[name] = self._run_for_image(op, frame, image_path)
            except Exception as e:
                logger.error("Operator %s failed on image: %s", name, e)
                results[name] = {"error": str(e)}
        return results

    def analyse_video(self, video_path: str) -> Dict[str, Any]:
        """Run shot-level operators (e.g. camera_movement) on a video file."""
        results: Dict[str, Any] = {}

        for name, op in self.operators.items():
            if op.granularity != "shot":
                continue
            try:
                results[name] = op.predict(video_path)
            except Exception as e:
                logger.error("Operator %s failed on video: %s", name, e)
                results[name] = {"error": str(e)}

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_for_image(
        self,
        op: BaseOperator,
        frame: np.ndarray,
        image_path: str,
    ) -> Dict[str, Any]:
        """Dispatch a single image to an operator based on its input_mode."""
        if op.input_mode == INPUT_FRAMES:
            return op.predict([frame], batch_size=self.config.batch_size)
        if op.input_mode == INPUT_FRAME_PATHS:
            return op.predict(image_path)
        if op.input_mode == INPUT_CUSTOM:
            return op.predict(image_path)
        raise OperatorInferenceError(
            f"Operator {op.name!r} has unsupported input_mode "
            f"{op.input_mode!r} for image analysis."
        )
