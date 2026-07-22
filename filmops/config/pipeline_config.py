"""Runtime configuration for the FilmOps pipeline.

Intentionally small: only parameters that vary by environment / call site
live here. Per-operator checkpoint paths are kept in
:class:`OperatorConfigs`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from filmops.config.operator_configs import OperatorConfigs


@dataclass
class FilmOpsConfig:
    """Global runtime configuration.

    Attributes:
        checkpoint_dir: Root directory containing all model weights.
        device: Inference device, e.g. ``"cuda"`` or ``"cpu"``.
        enabled_operators: Subset of operator names to enable.
            ``None`` enables every registered operator.
        batch_size: Per-batch frame count for frame-level operators.
        operator_configs: Per-operator checkpoint / weight paths.
    """

    checkpoint_dir: str = "./checkpoints"
    device: str = "cuda"
    enabled_operators: Optional[List[str]] = None

    batch_size: int = 8

    operator_configs: OperatorConfigs = field(default_factory=OperatorConfigs)

    def is_operator_enabled(self, name: str) -> bool:
        if self.enabled_operators is None:
            return True
        return name in self.enabled_operators

    def get_checkpoint_path(self, relative_path: str) -> str:
        """Resolve a path relative to :attr:`checkpoint_dir`."""
        return str(Path(self.checkpoint_dir) / relative_path)
