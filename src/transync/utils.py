"""Shared utilities for Transync."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def configure_logging(level: str | int = logging.INFO) -> None:
    """Configure a compact application logger."""

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def require_file(path: Path, description: str) -> Path:
    """Return ``path`` if it exists, otherwise raise a clear error."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def batched(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """Yield fixed-size batches from a sequence."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
