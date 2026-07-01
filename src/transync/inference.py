"""Convenience inference wrappers for Transync."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .translator import TransyncTranslator


@lru_cache(maxsize=4)
def get_translator(config_path: str | None = None) -> "TransyncTranslator":
    """Return a cached translator instance."""

    from .translator import TransyncTranslator

    return TransyncTranslator.load(config_path)


def translate_text(
    text: str,
    source_language: str,
    target_language: str,
    config_path: str | Path | None = None,
) -> str:
    """Translate one text string."""

    translator = get_translator(_cache_key(config_path))
    return translator.translate(text, source_language, target_language)


def translate_transync(
    text: str,
    sl: str,
    tl: str,
    config_path: str | Path | None = None,
) -> str:
    """Translate one text string using compact source and target codes."""

    return translate_text(text, sl, tl, config_path=config_path)


def translate_batch(
    texts: Sequence[str],
    sl: str,
    tl: str,
    batch_size: int = 32,
    config_path: str | Path | None = None,
) -> list[str]:
    """Translate a batch of texts."""

    translator = get_translator(_cache_key(config_path))
    return translator.translate_batch(texts, sl, tl, batch_size=batch_size)


def _cache_key(config_path: str | Path | None) -> str | None:
    if config_path is None:
        return None
    return str(Path(config_path).expanduser().resolve())
