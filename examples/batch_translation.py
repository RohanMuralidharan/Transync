"""Translate multiple sentences with Transync."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transync.inference import translate_batch


def main() -> None:
    sentences = [
        "Good morning.",
        "Thank you.",
        "See you tomorrow.",
    ]
    translations = translate_batch(
        sentences,
        sl="eng",
        tl="hin",
        batch_size=3,
    )

    for source, target in zip(sentences, translations):
        print(f"{source}\t{target}")


if __name__ == "__main__":
    main()
