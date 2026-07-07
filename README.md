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
  - ory
  - pan
  - san
  - sat
  - snd
  - tam
  - tel
  - urd
  - eng
license: mit
library_name: transformers
pipeline_tag: text2text-generation
tags:
  - translation
  - multilingual
  - indic
  - mbart
  - seq2seq
  - huggingface
---

# Transync

[![Hugging Face Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/RohanMuralidharan/transync)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

## Model Overview

Transync is a multilingual translation model based on the MBart architecture, designed for offline neural machine translation. This repository provides inference capabilities for translating between 50+ languages including all major Indian languages (Hindi, Telugu, Tamil, Bengali, Gujarati, Marathi, Punjabi, Urdu, and more).

## Features

- ✅ **Multilingual Support**: 50+ languages including all major Indian languages
- ✅ **High-Quality Translations**: Based on the MBart architecture
- ✅ **Offline Operation**: Works without internet connection
- ✅ **Efficient Inference**: Optimized for both CPU and GPU
- ✅ **Batch Processing**: Supports batch translation for efficiency
- ✅ **CLI Interface**: Command-line interface for easy usage

## Supported Languages

The model supports 50+ languages including:

### Indian Languages
- Assamese (asm)
- Bengali (ben)
- Bodo (brx)
- Dogri (doi)
- Konkani (gom)
- Gujarati (guj)
- Hindi (hin)
- Kannada (kan)
- Kashmiri (kas)
- Maithili (mai)
- Malayalam (mal)
- Marathi (mar)
- Manipuri (mni)
- Nepali (npi)
- Odia (ory)
- Punjabi (pan)
- Sanskrit (san)
- Santali (sat)
- Sindhi (snd)
- Tamil (tam)
- Telugu (tel)
- Urdu (urd)

### Other Languages
- Arabic (ara)
- Czech (ces)
- German (deu)
- Spanish (spa)
- Estonian (est)
- Finnish (fin)
- French (fra)
- Hebrew (heb)
- Croatian (hrv)
- Indonesian (ind)
- Italian (ita)
- Japanese (jpn)
- Georgian (kat)
- Kazakh (kaz)
- Khmer (khm)
- Korean (kor)
- Lithuanian (lit)
- Latvian (lav)
- Macedonian (mkd)
- Mongolian (mon)
- Burmese (mya)
- Dutch (nld)
- Polish (pol)
- Pashto (pus)
- Portuguese (por)
- Romanian (ron)
- Russian (rus)
- Sinhala (sin)
- Slovak (slk)
- Swedish (swe)
- Swahili (swa)
- Thai (tha)
- Tagalog (tgl)
- Turkish (tur)
- Ukrainian (ukr)
- Vietnamese (vie)
- Xhosa (xho)
- Chinese (zho)
- Azerbaijani (aze)
- Persian (fas)
- Galician (glg)
- Afrikaans (afr)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from transync_inference import translate_onemt

# Translate English to Hindi
result = translate_onemt("Hello, how are you?", "eng", "hin")
print(result)  # नमस्ते, आप कैसे हैं?
```

## Python Example

```python
from transync_inference import translate_onemt, translate_batch

# Single translation
translation = translate_onemt(
    text="Good morning!",
    source_lang="eng",
    target_lang="hin"
)
print(translation)

# Batch translation
texts = ["Hello", "How are you?", "Goodbye"]
results = translate_batch(
    texts=texts,
    source_lang="eng",
    target_lang="hin"
)
print(results)
```

## Transformers Example

```python
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Load model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("RohanMuralidharan/transync")
tokenizer = MBart50Tokenizer.from_pretrained("RohanMuralidharan/transync")

# Set source and target languages
tokenizer.src_lang = "en_XX"
target_lang = "hi_IN"

# Encode input text
input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids

# Generate translation
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )

# Decode translation
translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated)
```

## CLI Example

```bash
# Translate single text
python transync_inference.py eng hin "Hello, how are you?"

# Batch translation from file
python transync_inference.py --batch eng hin -f input.txt -o output.txt
```

## Repository Structure

```
transync/
├── config.json                 # Model configuration
├── generation_config.json      # Generation configuration
├── pytorch_model.bin           # Model weights
├── sentencepiece.bpe.model     # SentencePiece tokenizer model
├── tokenizer_config.json       # Tokenizer configuration
├── special_tokens_map.json     # Special tokens mapping
├── README.md                   # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Dependencies
└── transync_inference.py       # Inference code
```

## Model Architecture

This model is built on the MBart (Multilingual BART) architecture, which is a sequence-to-sequence model pre-trained on multilingual data. It leverages the power of BART's bidirectional encoder and autoregressive decoder for translation tasks.

## Tokenizer

The model uses SentencePiece tokenizer for subword tokenization. The tokenizer is compatible with the MBart50 tokenizer format and supports 50+ languages.

## Intended Uses

- Offline multilingual translation
- Research and educational purposes
- Integration into translation applications
- Batch processing of text translations

## Out-of-Scope Uses

- Training new models from scratch (this is a pre-trained model)
- Commercial use without proper licensing (please check license terms)
- Use in production systems without proper testing and validation

## Limitations

- Translation quality depends on training data coverage
- May struggle with domain-specific terminology
- Performance varies between languages
- Requires sufficient computational resources for optimal performance

## Ethical Considerations

This model is intended for educational and research purposes. Users should be aware of the following considerations:

- Translation accuracy may vary depending on language pair and domain
- The model should not be used to generate misleading or harmful content
- Users should respect copyright and intellectual property rights when using translations
- The model's training data sources and limitations should be understood

## Hardware Requirements

- CPU: Minimum 4GB RAM, Recommended 8GB+
- GPU: NVIDIA GPU with CUDA support (recommended for faster inference)
- Storage: At least 2.4GB of storage space for model weights

## Performance Notes

- CPU inference is suitable for small-scale tasks
- GPU acceleration significantly improves translation speed
- Batch processing is more efficient than individual translations
- Translation quality may vary between language pairs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This model is based on the MBart architecture and uses pre-trained weights from the Hugging Face ecosystem. The model weights are not included in this repository and must be downloaded separately from the Hugging Face Hub.
**Option 1: Direct Download**
1. Go to: https://huggingface.co/RohanMuralidharan/Transync
2. Download `pytorch_model.bin` (2.44 GB)
3. Place it in the project root directory

**Option 2: Using Python**
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='RohanMuralidharan/Transync',
    filename='pytorch_model.bin',
    local_dir='.'
)
```

**Option 3: Using huggingface-cli**
```bash
huggingface-cli download RohanMuralidharan/Transync pytorch_model.bin
```

---

### Command Line Usage

```bash
python transync_inference.py eng hin "Hello, how are you?"
```

---

## Supported Languages

| Short Code | Language | Script | MBart Code |
|---|---|---|---|
| `eng` | English | Latin | `en_XX` |
| `hin` | Hindi | Devanagari | `hi_IN` |
| `tel` | Telugu | Telugu | `te_IN` |
| `tam` | Tamil | Tamil | `ta_IN` |
| `mal` | Malayalam | Malayalam | `ml_IN` |
| `kan` | Kannada | Kannada | `kn_IN` |
| `ben` | Bengali | Bengali | `bn_IN` |
| `guj` | Gujarati | Gujarati | `gu_IN` |
| `mar` | Marathi | Devanagari | `mr_IN` |
| `pan` | Punjabi | Gurmukhi | `pa_IN` |
| `urd` | Urdu | Arabic | `ur_PK` |
| `asm` | Assamese | Bengali | `as_IN` |
| `npi` | Nepali | Devanagari | `ne_NP` |
| `ory` | Odia | Odia | `or_IN` |
| `san` | Sanskrit | Devanagari | `sa_IN` |
| `mai` | Maithili | Devanagari | `mai_IN` |
| `brx` | Bodo | Devanagari | `brx_IN` |
| `doi` | Dogri | Devanagari | `doi_IN` |
| `gom` | Konkani | Devanagari | `gom_IN` |
| `mni` | Meitei | Bengali | `mni_IN` |
| `sat` | Santali | Ol Chiki | `sat_IN` |
| `kas` | Kashmiri | Arabic | `ks_IN` |
| `snd` | Sindhi | Arabic | `sd_IN` |

---

## How It Works

Transync uses standard MBart tokenization with SentencePiece:

```
source text
  → Add language tag: [en_XX] Hello, how are you?
  → SentencePiece encoding → subword pieces
  → MBart encoder
```

The model uses the standard MBart50 tokenizer with language codes for high-quality multilingual translation.

---

## Repository Files

| File | Description |
|---|---|
| `pytorch_model.bin` | Model weights (~2.4 GB) |
| `config.json` | Model architecture config |
| `generation_config.json` | Generation parameters |
| `sentencepiece.bpe.model` | SentencePiece tokenizer |
| `tokenizer_config.json` | Tokenizer config |
| `special_tokens_map.json` | Special tokens mapping |
| `transync_inference.py` | Main inference script |

---

## Verification

Test the model with:
```bash
python transync_inference.py eng hin "Hello, how are you?"
```

Expected output: `नमस्ते, आप कैसे हैं?`

---

## License

MIT

