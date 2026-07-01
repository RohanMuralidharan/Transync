"""Custom SentencePiece plus fairseq dictionary tokenizer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import sentencepiece as spm

from .utils import load_json, require_file


class FairseqSentencePieceTokenizer:
    """Encode/decode text with the model's fairseq token ID ordering."""

    def __init__(self, sentencepiece_model_path: Path, dictionary_path: Path) -> None:
        self.sentencepiece_model_path = require_file(
            sentencepiece_model_path,
            "SentencePiece model",
        )
        self.dictionary_path = require_file(dictionary_path, "fairseq dictionary")
        self.processor = spm.SentencePieceProcessor(
            model_file=str(self.sentencepiece_model_path)
        )

        dictionary = load_json(self.dictionary_path)
        self.source_symbol_to_id = {
            str(symbol): int(token_id)
            for symbol, token_id in dictionary["src"].items()
        }
        self.target_id_to_symbol = {
            int(token_id): str(symbol)
            for symbol, token_id in dictionary["tgt"].items()
        }

        special_tokens = dictionary["special"]
        self.eos_id = int(special_tokens["eos"])
        self.pad_id = int(special_tokens["pad"])
        self.bos_id = int(special_tokens["bos"])
        self.unk_id = int(special_tokens["unk"])

    def encode(self, text: str, source_flores: str, target_flores: str) -> list[int]:
        """Encode source text using Transync's fairseq vocabulary IDs."""

        tagged_text = f"###{source_flores}-to-{target_flores}### {text}"
        pieces = self.processor.encode(tagged_text, out_type=str)
        token_ids = [
            self.source_symbol_to_id.get(piece, self.unk_id) for piece in pieces
        ]
        token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode generated target token IDs into text."""

        special_ids = {self.bos_id, self.pad_id, self.eos_id}
        pieces = [
            self.target_id_to_symbol.get(int(token_id), "<unk>")
            for token_id in token_ids
            if int(token_id) not in special_ids
        ]
        return self.processor.decode(pieces).strip()
