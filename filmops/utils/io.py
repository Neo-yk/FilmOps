"""Generic file I/O helpers."""

import json
from pathlib import Path
from typing import Any


def dump_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    """Dump ``obj`` to ``path`` as UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
