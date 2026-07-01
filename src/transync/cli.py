"""Command-line interface for Transync."""

from __future__ import annotations

import argparse
from pathlib import Path

from .inference import get_translator
from .utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the Transync CLI parser."""

    parser = argparse.ArgumentParser(
        prog="transync",
        description="Translate text with the Transync Indic multilingual model.",
    )
    parser.add_argument("-s", "--source", required=True, help="Source language code")
    parser.add_argument("-t", "--target", required=True, help="Target language code")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a Transync JSON config file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for file translation",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Python logging level",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=None,
        help="Optional output file",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Text to translate")
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="UTF-8 text file with one segment per line",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Transync CLI."""

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    config_key = str(args.config.expanduser().resolve()) if args.config else None
    translator = get_translator(config_key)

    if args.text is not None:
        outputs = [translator.translate(args.text, args.source, args.target)]
    else:
        inputs = args.input_file.read_text(encoding="utf-8").splitlines()
        outputs = translator.translate_batch(
            inputs,
            args.source,
            args.target,
            batch_size=args.batch_size,
        )

    output_text = "\n".join(outputs)
    if args.output_file:
        args.output_file.write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
