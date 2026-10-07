"""Small shared helpers (logging and JSON I/O)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _default(obj: Any) -> Any:
    """Make numpy scalars/arrays JSON serialisable."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Cannot serialise {type(obj)}")


def save_json(data: Any, path: Path) -> None:
    """Write ``data`` to ``path`` as pretty JSON, creating folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_default), encoding="utf-8")


def load_json(path: Path) -> Any:
    """Read JSON from ``path``; raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))
