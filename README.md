# Transync — Indic Multilingual Translation

Transync is a powerful multilingual translation model supporting **50+ languages** including all major Indian languages (Hindi, Telugu, Tamil, Bengali, Gujarati, Marathi, Punjabi, Urdu, and more). Built on Facebook's MBart architecture, it provides high-quality translations for Indic languages.
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
## Installation

```bash
pip install -r requirements.txt
```

### Download Model Weights

The model weights are not included in the repository. Download them from:

**Option 1: Direct Download**
1. Go to: https://huggingface.co/RohanMuralidharan/Transync
2. Download `pytorch_model.bin` 
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

## Verification

Test the model with:
```bash
python transync_inference.py eng hin "Hello, how are you?"
```

---

## License

MIT

