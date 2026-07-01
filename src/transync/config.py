"""Runtime configuration for Transync."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import load_json

DEFAULT_CONFIG_RELATIVE_PATH = Path("configs") / "transync.json"
DEFAULT_HF_REPO = "RohanMuralidharan/Transync"


@dataclass(frozen=True)
class GenerationSettings:
    """Text generation parameters that mirror the original inference script."""

    beam_size: int = 5
    max_new_tokens: int = 256
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenerationSettings":
        """Build generation settings from a JSON dictionary."""

        if data is None:
            return cls()

        return cls(
            beam_size=int(data.get("beam_size", cls.beam_size)),
            max_new_tokens=int(data.get("max_new_tokens", cls.max_new_tokens)),
            early_stopping=bool(data.get("early_stopping", cls.early_stopping)),
            no_repeat_ngram_size=int(
                data.get("no_repeat_ngram_size", cls.no_repeat_ngram_size)
            ),
            repetition_penalty=float(
                data.get("repetition_penalty", cls.repetition_penalty)
            ),
        )


@dataclass(frozen=True)
class TransyncConfig:
    """Resolved Transync runtime configuration."""

    hf_repo: str
    model_dir: Path
    sentencepiece_model: Path
    dictionary_path: Path
    device: str
    generation: GenerationSettings
    long_text_word_threshold: int = 200
    chunk_word_size: int = 100

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "TransyncConfig":
        """Load and resolve a Transync configuration file."""

        config_path = _resolve_config_path(path)
        data = load_json(config_path)
        base_dir = _config_base_dir(config_path)

        return cls(
            hf_repo=str(data.get("hf_repo", DEFAULT_HF_REPO)),
            model_dir=_resolve_path(data.get("model_dir", "models/transync"), base_dir),
            sentencepiece_model=_resolve_path(
                data.get("sentencepiece_model", "tokenizer/transync_spm.model"),
                base_dir,
            ),
            dictionary_path=_resolve_path(
                data.get("dictionary_path", "tokenizer/fairseq_dict.json"),
                base_dir,
            ),
            device=str(data.get("device", "auto")),
            generation=GenerationSettings.from_dict(data.get("generation")),
            long_text_word_threshold=int(data.get("long_text_word_threshold", 200)),
            chunk_word_size=int(data.get("chunk_word_size", 100)),
        )


def find_project_root() -> Path:
    """Find the repository root by looking for the default config file."""

    starts = (Path.cwd(), Path(__file__).resolve())
    seen: set[Path] = set()

    for start in starts:
        current = start if start.is_dir() else start.parent
        for candidate in (current, *current.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / DEFAULT_CONFIG_RELATIVE_PATH).is_file():
                return candidate.resolve()

    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """Return the default configuration path."""

    env_path = os.getenv("TRANSYNC_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return find_project_root() / DEFAULT_CONFIG_RELATIVE_PATH


def _resolve_config_path(path: str | Path | None) -> Path:
    if path is None:
        resolved = default_config_path()
    else:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            cwd_candidate = (Path.cwd() / resolved).resolve()
            root_candidate = (find_project_root() / resolved).resolve()
            resolved = cwd_candidate if cwd_candidate.is_file() else root_candidate
        resolved = resolved.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"Missing Transync configuration file: {resolved}")
    return resolved


def _config_base_dir(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
