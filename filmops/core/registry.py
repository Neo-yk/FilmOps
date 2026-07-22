"""Operator registry.

Each operator class self-registers via the :func:`register_operator`
decorator. The Pipeline iterates over the registry instead of hard-coding
per-operator branches.
"""

from typing import Dict, List, Type

from filmops.core.base import BaseOperator
from filmops.core.exceptions import OperatorNotFoundError

_REGISTRY: Dict[str, Type[BaseOperator]] = {}


def register_operator(name: str):
    """Class decorator that registers an operator under ``name``.

    Usage::

        @register_operator("shot_scale")
        class ShotScaleOperator(BaseOperator):
            ...
    """

    def _wrap(cls: Type[BaseOperator]) -> Type[BaseOperator]:
        if not issubclass(cls, BaseOperator):
            raise TypeError(
                f"{cls.__name__} must subclass BaseOperator to be registered."
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def resolve_operator(name: str) -> Type[BaseOperator]:
    """Return the operator class registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise OperatorNotFoundError(
            f"Operator {name!r} is not registered. "
            f"Available: {sorted(_REGISTRY)}"
        ) from exc


def list_operators() -> List[str]:
    """Return the sorted list of registered operator names."""
    return sorted(_REGISTRY)
