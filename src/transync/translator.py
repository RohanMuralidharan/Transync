"""High-level translation interface for Transync."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
from transformers import MBartForConditionalGeneration

from .config import TransyncConfig
from .languages import resolve_language_code
from .tokenizer import FairseqSentencePieceTokenizer
from .utils import batched

LOGGER = logging.getLogger(__name__)


class TransyncTranslator:
    """Load the Transync model and run translation."""

    def __init__(
        self,
        config: TransyncConfig,
        tokenizer: FairseqSentencePieceTokenizer,
        model: MBartForConditionalGeneration,
        device: torch.device,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "TransyncTranslator":
        """Load tokenizer, model, and runtime configuration."""

        config = TransyncConfig.from_file(config_path)
        device = _resolve_device(config.device)
        tokenizer = FairseqSentencePieceTokenizer(
            sentencepiece_model_path=config.sentencepiece_model,
            dictionary_path=config.dictionary_path,
        )
        model_source = _model_source(config)

        LOGGER.info("Loading model from %s", model_source)
        model = MBartForConditionalGeneration.from_pretrained(str(model_source))
        model.eval().to(device)
        LOGGER.info(
            "Transync ready on %s with %s parameters",
            device,
            f"{model.num_parameters():,}",
        )
        return cls(config=config, tokenizer=tokenizer, model=model, device=device)

    def encode(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> list[int]:
        """Encode text for a source-target language pair."""

        source_flores = resolve_language_code(source_language)
        target_flores = resolve_language_code(target_language)
        return self.tokenizer.encode(text, source_flores, target_flores)

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode generated token IDs into text."""

        return self.tokenizer.decode(token_ids)

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        """Translate a single text string."""

        source_flores = resolve_language_code(source_language)
        target_flores = resolve_language_code(target_language)

        if len(text.split()) > self.config.long_text_word_threshold:
            outputs = []
            for part in split_into_word_chunks(text, self.config.chunk_word_size):
                translated = self._translate_encoded(
                    [self.tokenizer.encode(part, source_flores, target_flores)]
                )
                outputs.append(translated[0] if translated and translated[0] else part)
            return " ".join(outputs).strip()

        translated = self._translate_encoded(
            [self.tokenizer.encode(text, source_flores, target_flores)]
        )
        return (translated[0] if translated and translated[0] else text).strip()

    def translate_batch(
        self,
        texts: Sequence[str],
        source_language: str,
        target_language: str,
        batch_size: int = 32,
    ) -> list[str]:
        """Translate a sequence of texts."""

        source_flores = resolve_language_code(source_language)
        target_flores = resolve_language_code(target_language)
        results: list[str] = []

        for batch in batched(texts, batch_size):
            encoded = [
                self.tokenizer.encode(text, source_flores, target_flores)
                for text in batch
            ]
            results.extend(self._translate_encoded(encoded))
            LOGGER.info("Translated %s/%s texts", len(results), len(texts))

        return results

    def _translate_encoded(self, encoded_batch: Sequence[Sequence[int]]) -> list[str]:
        if not encoded_batch:
            return []

        max_length = max(len(token_ids) for token_ids in encoded_batch)
        input_ids = torch.full(
            (len(encoded_batch), max_length),
            self.tokenizer.pad_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)

        for index, token_ids in enumerate(encoded_batch):
            length = len(token_ids)
            input_ids[index, :length] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[index, :length] = 1

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        generation = self.config.generation

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=self.tokenizer.eos_id,
                num_beams=generation.beam_size,
                max_new_tokens=generation.max_new_tokens,
                early_stopping=generation.early_stopping,
                no_repeat_ngram_size=generation.no_repeat_ngram_size,
                repetition_penalty=generation.repetition_penalty,
            )

        return [self.tokenizer.decode(output.tolist()) for output in generated]


def split_into_word_chunks(text: str, chunk_size: int = 100) -> list[str]:
    """Split long text into fixed-size word chunks."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    words = text.split()
    return [
        " ".join(words[start : start + chunk_size])
        for start in range(0, len(words), chunk_size)
    ]


def _resolve_device(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _model_source(config: TransyncConfig) -> Path | str:
    if config.model_dir.is_dir():
        return config.model_dir

    LOGGER.warning(
        "Local model directory %s was not found; falling back to Hugging Face repo %s",
        config.model_dir,
        config.hf_repo,
    )
    return config.hf_repo
