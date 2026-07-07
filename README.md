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
tags:
- translation
- multilingual
- indic
- mbart
- seq2seq
license: mit
---

# Transync — Indic Multilingual Translation

Transync is a powerful multilingual translation model supporting **50+ languages** including all major Indian languages (Hindi, Telugu, Tamil, Bengali, Gujarati, Marathi, Punjabi, Urdu, and more). Built on Facebook's MBart architecture, it provides high-quality translations for Indic languages.

---

## Installation

```bash
pip install -r requirements.txt
```

### Download Model Weights

The model weights (~2.4 GB) are not included in the repository. Download them from:

**Option 1: Direct Download**
1. Go to: https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt
2. Download `pytorch_model.bin` (2.44 GB)
3. Place it in the project root directory

**Option 2: Using Python**
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/mbart-large-50-one-to-many-mmt',
    filename='pytorch_model.bin',
    local_dir='.'
)
```

**Option 3: Using huggingface-cli**
```bash
huggingface-cli download facebook/mbart-large-50-one-to-many-mmt pytorch_model.bin
```

---

## Quick Start

```python
from transync_inference import translate_onemt, translate_batch

# Single sentence — English → Hindi
print(translate_onemt("Hello, how are you?", "eng", "hin"))
# Output: नमस्ते, आप कैसे हैं?

# Code-mixed Hindi → Telugu
print(translate_onemt("मुझे meeting attend करनी है।", "hin", "tel"))
# Output: నాకు మీటింగ్‌కు హాజరు కావాలి.

# Telugu → English
print(translate_onemt("నమస్కారం, మీరు ఎలా ఉన్నారు?", "tel", "eng"))
# Output: Hello, how are you?

# Tamil → Hindi
print(translate_onemt("இன்று வானிலை நன்றாக உள்ளது.", "tam", "hin"))
# Output: आज मौसम अच्छा है।

# Batch translation
sentences = ["Good morning.", "Thank you.", "See you tomorrow."]
results   = translate_batch(sentences, sl="eng", tl="hin", batch_size=32)
for src, tgt in zip(sentences, results):
    print(f"{src}  →  {tgt}")
```

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
| `fine_tune_example.py` | Example fine-tuning script |

---

## Model Details

- **Base Model**: facebook/mbart-large-50-one-to-many-mmt
- **Parameters**: ~610M
- **Languages**: 50+ (including all major Indian languages)
- **License**: MIT

---

## Making It Your Own (Fine-Tuning)

To create your own custom model, fine-tune on your data:

### Step 1: Prepare Your Training Data
Create a CSV or JSON file with translation pairs:
```csv
source,target
"Hello, how are you?","नमस्ते, आप कैसे हैं?"
"Good morning","सुप्रभात"
"Thank you","धन्यवाद"
```

### Step 2: Fine-Tune
Use the provided `fine_tune_example.py` script. You'll need:
- At least 1,000-10,000 translation pairs for good results
- A GPU with 8GB+ VRAM (or use Google Colab)
- 1-3 hours of training time

### Step 3: Replace Model Weights
After fine-tuning, replace `pytorch_model.bin` with your fine-tuned model.

---

## Verification

Test the model with:
```bash
python transync_inference.py eng hin "Hello, how are you?"
```

Expected output: `नमस्ते, आप कैसे हैं?`

---

## Citation

If you use this model, please cite the original MBart paper:

```bibtex
@article{liu2020mbart,
  title={Multilingual Denoising Pre-training for Neural Machine Translation},
  author={Liu, Yinhan and Gu, Jiatao and Goyal, Naman and Li, Xian and Edunov, Sergey and Ghazvininejad, Marjan and Lewis, Mike and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2001.08210},
  year={2020}
}
```

---

## License

MIT

## Acknowledgments

- Base model: [facebook/mbart-large-50-one-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt)