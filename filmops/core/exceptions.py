"""FilmOps exception hierarchy."""


class FilmOpsError(Exception):
    """Base class for all FilmOps errors."""


class OperatorNotFoundError(FilmOpsError):
    """Raised when an operator name cannot be resolved in the registry."""


class OperatorLoadError(FilmOpsError):
    """Raised when an operator fails to load its model or weights."""


class OperatorInferenceError(FilmOpsError):
    """Raised when an operator fails during inference."""
