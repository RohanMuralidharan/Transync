---
language:
  - asm
  - ben
  - brx
  - doi
  - gom
  - guj
  - hin
  - kan
  - kas
  - mai
  - mal
  - mar
  - mni
  - npi
  - ori
  - pan
  - san
  - sat
  - snd
  - tam
  - tel
  - urd
  - eng
tags:
  - translation
  - multilingual
  - indic
  - mbart
  - fairseq
  - seq2seq
license: mit
library_name: transformers
pipeline_tag: translation
arxiv: https://arxiv.org/abs/2412.04351
---

# Transync

Transync is a production-oriented Indic multilingual neural machine translation
project. It provides a clean Python package, CLI, examples, configuration files,
and documentation while preserving the model behavior, custom tokenizer
requirements, citation, and license metadata.

The bundled model is a roughly 1.1B parameter multilingual sequence-to-sequence
translation model wrapped with `MBartForConditionalGeneration`. It supports
translation across English and multiple Indian subcontinent languages using a
custom SentencePiece plus fairseq dictionary tokenization pipeline.

## Features

- Local-first inference from bundled `safetensors` weights.
- Custom fairseq ID mapping preserved for correct tokenization.
- Python API for single sentence and batch translation.
- CLI for shell workflows and file-based batch translation.
- JSON runtime configuration for model paths, tokenizer paths, device, and
  generation settings.
- Professional repository layout for GitHub and Hugging Face.
- Transync model naming across code, configuration, tokenizer files, and docs.

## Architecture

Transync keeps the same model and generation path as the original inference
script:

```text
input text
  -> language tag: ###src_flores-to-tgt_flores### {text}
  -> SentencePiece segmentation
  -> fairseq dictionary token ID lookup
  -> MBartForConditionalGeneration
  -> fairseq target ID decoding
  -> SentencePiece text decoding
```

Important: standard `AutoTokenizer` or MBart tokenizers should not be used with
these artifacts. The fairseq dictionary order differs from the native
SentencePiece vocabulary order, so using the wrong tokenizer will produce
incorrect token IDs and unreliable output.

## Installation

Create an environment with Python 3.10 or newer, then install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

For GPU inference, install a PyTorch build that matches your CUDA runtime before
installing the rest of the dependencies.

## Quick Start

```python
from transync.inference import translate_text, translate_batch

print(translate_text("Hello, how are you?", "eng", "hin"))

sentences = ["Good morning.", "Thank you.", "See you tomorrow."]
translations = translate_batch(sentences, sl="eng", tl="tel", batch_size=3)
for source, target in zip(sentences, translations):
    print(source, "->", target)
```

CLI:

```bash
transync --source eng --target hin --text "Hello, how are you?"
transync --source eng --target tel --input-file examples/input.txt --output-file output.txt
```

You can also run from a source checkout without installing the console script:

```bash
python scripts/translate.py --source eng --target hin --text "Good morning."
```

## Repository Structure

```text
Transync/
|-- README.md
|-- LICENSE
|-- NOTICE
|-- requirements.txt
|-- pyproject.toml
|-- setup.py
|-- configs/
|   `-- transync.json
|-- models/
|   `-- transync/
|       |-- config.json
|       |-- generation_config.json
|       `-- model.safetensors
|-- tokenizer/
|   |-- fairseq_dict.json
|   `-- transync_spm.model
|-- src/
|   `-- transync/
|-- scripts/
|-- examples/
|-- docs/
|-- tests/
`-- assets/
```

## Supported Languages

Accepted language codes are mapped to FLORES-style tags before tokenization.

| Code | Language | Model tag |
|---|---|---|
| `asm` | Assamese | `asm_Beng` |
| `ben` | Bengali | `ben_Beng` |
| `brx` | Bodo | `brx_Deva` |
| `doi` | Dogri | `doi_Deva` |
| `gom` | Konkani | `gom_Deva` |
| `guj` | Gujarati | `guj_Gujr` |
| `hin` | Hindi | `hin_Deva` |
| `kan` | Kannada | `kan_Knda` |
| `kas` | Kashmiri | `kas_Arab` |
| `kas_deva` | Kashmiri | `kas_Deva` |
| `mai` | Maithili | `mai_Deva` |
| `mal` | Malayalam | `mal_Mlym` |
| `mar` | Marathi | `mar_Deva` |
| `mni` | Meitei | `mni_Beng` |
| `mni_mtei` | Meitei | `mni_Mtei` |
| `npi` | Nepali | `npi_Deva` |
| `ory` | Odia | `ory_Orya` |
| `pan` | Punjabi | `pan_Guru` |
| `san` | Sanskrit | `san_Deva` |
| `sat` | Santali | `sat_Olck` |
| `snd` | Sindhi | `snd_Arab` |
| `snd_deva` | Sindhi | `snd_Deva` |
| `tam` | Tamil | `tam_Taml` |
| `tel` | Telugu | `tel_Telu` |
| `urd` | Urdu | `urd_Arab` |
| `eng` | English | `eng_Latn` |

Aliases such as `ori`, `odi`, `pun`, `ban`, `kas_arab`, `mni_beng`,
`snd_arab`, and `urd_arab` are also accepted.

## Performance

The model has approximately 1.1B parameters and is best suited to GPU inference.
CPU inference works for small requests but can be slow. Default generation
settings match the original script: beam size 5, maximum 256 new tokens,
no-repeat n-gram size 3, repetition penalty 1.3, and EOS as the decoder start
token.

## Applications

- Multilingual Indic translation prototypes.
- Batch translation pipelines for aligned sentence or paragraph files.
- Local research experiments where remote translation APIs are not desired.
- Fine-grained evaluation of custom Indic tokenization and generation behavior.

## Limitations

- The model requires the bundled custom tokenizer artifacts.
- Long text is split by word count, so document-level context is not preserved.
- Translation quality varies by language pair, domain, and script.
- The project does not add safety filtering or factuality guarantees.
- The included model artifacts are large; use Git LFS for repository hosting.

## Configuration

Runtime options live in `configs/transync.json`. Set `TRANSYNC_CONFIG` or pass
`--config` to use a different configuration file. Relative artifact paths are
resolved from the repository root when the config is stored in `configs/`.

## Citation

If you use Transync in research or production reports, cite the project:

```bibtex
@software{transync2026,
      title={Transync: Indic Multilingual Neural Machine Translation},
      author={Transync Contributors},
      year={2026},
      url={https://github.com/RohanMuralidharan/Transync},
}
```

## License

Transync is distributed under the MIT License. See `LICENSE`.

See `NOTICE` and `docs/attribution.md` for redistribution notes and project
metadata.

## Acknowledgements

Transync acknowledges the open-source machine learning ecosystem that makes
local multilingual translation possible. The tokenizer warning, fairseq
dictionary requirement, and generation conventions are preserved because they
are essential to correct inference.

## Future Work

- Add benchmark scripts for common CPU and GPU configurations.
- Add optional quantized inference profiles.
- Add document-aware chunking and post-processing utilities.
- Publish a Hugging Face model card with validated examples per language pair.
