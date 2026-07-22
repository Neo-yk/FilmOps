"""Abstract base class for all FilmOps operators.

Each operator declares two things up-front so the Pipeline can dispatch
without hard-coded special cases:

* ``granularity`` — ``"frame"`` for per-frame classifiers, ``"shot"`` for
  shot-level operators that consume a whole clip.
* ``input_mode`` — what shape of input ``predict()`` expects when invoked
  by the Pipeline. See ``filmops.core.types`` for the constants.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from filmops.core.types import INPUT_FRAMES, OperatorInput


class BaseOperator(ABC):
    """Abstract base class for all FilmOps operators.

    Subclasses must implement :meth:`load` and :meth:`predict`. They should
    also override :attr:`name`, :attr:`granularity`, and :attr:`input_mode`.
    """

    #: Unique operator identifier (snake_case). Matches the registry key.
    name: str = "base"

    #: ``"frame"`` (per-frame classifier) or ``"shot"`` (per-clip classifier).
    granularity: str = "frame"

    #: What the Pipeline should feed to :meth:`predict`. One of the constants
    #: in :mod:`filmops.core.types` (``INPUT_FRAMES``,
    #: ``INPUT_FRAME_PATHS``, ``INPUT_VIDEO_PATH``, ``INPUT_CUSTOM``).
    input_mode: str = INPUT_FRAMES

    @abstractmethod
    def load(self, **kwargs) -> None:
        """Load model weights and initialize the operator."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: OperatorInput, **kwargs) -> Dict[str, Any]:
        """Run inference on the given inputs.

        Returns:
            A dict containing at minimum a ``labels`` key (list of strings).
            May also include ``per_frame``, ``raw``, ``confidence`` etc.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"granularity={self.granularity!r}, "
            f"input_mode={self.input_mode!r})"
        )
