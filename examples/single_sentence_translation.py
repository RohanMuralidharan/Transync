"""Translate a single sentence with Transync."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transync.inference import translate_text


def main() -> None:
    translation = translate_text(
        "Hello, how are you?",
        source_language="eng",
        target_language="hin",
    )
    print(translation)


if __name__ == "__main__":
    main()
