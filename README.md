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
arxiv: https://arxiv.org/pdf/2412.04351
---

# OneNMT v3b — Indic Multilingual Neural Machine Translation

OneNMT v3b is a **1.1B parameter** multilingual seq2seq translation model supporting **36 × 36 language pairs** across 35 Indian languages and English. It is based on a deep `transformer_18_18` architecture (18 encoder + 18 decoder layers) trained with fairseq and converted to the HuggingFace MBart format.

> ⚠️ **Tokenization Warning:** OneNMT v3b uses a custom fairseq dictionary (`fairseq_dict.json`) for token ID mapping. Using `AutoTokenizer` or any standard MBart tokenizer will produce wrong IDs and garbled output. Always use the provided `hf_inference.py` or the `translate_onemt()` wrapper.

---

## Architecture

| Property | Value |
|---|---|
| Architecture | `transformer_18_18` (MBartForConditionalGeneration wrapper) |
| Encoder / Decoder layers | 18 + 18 |
| Hidden size | 1024 |
| Attention heads | 16 |
| Parameters | ~1.1B |
| Positional embeddings | Sinusoidal (not learned), max 256 positions |
| Normalization | Pre-norm (`normalize_before=True`) |
| Vocabulary | 65,400 tokens (fairseq dictionary ordering) |
| Language tag format | `###src_flores-to-tgt_flores### {text}` |
| Decoder start token | EOS (id=2) — fairseq convention |

---

## Supported Languages

| Short Code | Language | Script | FLORES-200 Code |
|---|---|---|---|
| `eng` | English | Latin | `eng_Latn` |
| `hin` | Hindi | Devanagari | `hin_Deva` |
| `tel` | Telugu | Telugu | `tel_Telu` |
| `tam` | Tamil | Tamil | `tam_Taml` |
| `mal` | Malayalam | Malayalam | `mal_Mlym` |
| `kan` | Kannada | Kannada | `kan_Knda` |
| `ben` | Bengali | Bengali | `ben_Beng` |
| `guj` | Gujarati | Gujarati | `guj_Gujr` |
| `mar` | Marathi | Devanagari | `mar_Deva` |
| `pan` | Punjabi | Gurmukhi | `pan_Guru` |
| `urd` | Urdu | Arabic | `urd_Arab` |
| `asm` | Assamese | Bengali | `asm_Beng` |
| `npi` | Nepali | Devanagari | `npi_Deva` |
| `ory` | Odia | Odia | `ory_Orya` |
| `san` | Sanskrit | Devanagari | `san_Deva` |
| `mai` | Maithili | Devanagari | `mai_Deva` |
| `brx` | Bodo | Devanagari | `brx_Deva` |
| `doi` | Dogri | Devanagari | `doi_Deva` |
| `gom` | Konkani | Devanagari | `gom_Deva` |
| `mni` | Meitei | Bengali | `mni_Beng` |
| `sat` | Santali | Ol Chiki | `sat_Olck` |
| `kas` | Kashmiri | Arabic | `kas_Arab` |
| `snd` | Sindhi | Arabic | `snd_Arab` |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from hf_inference import translate_onemt, translate_batch

# Single sentence — English → Hindi
print(translate_onemt("Hello, how are you?", "eng", "hin"))
# Expected: नमस्ते, आप कैसे हैं?

# Code-mixed Hindi → Telugu
print(translate_onemt("मुझे meeting attend करनी है।", "hin", "tel"))
# Expected: నాకు మీటింగ్‌కు హాజరు కావాలి.

# Telugu → English
print(translate_onemt("నమస్కారం, మీరు ఎలా ఉన్నారు?", "tel", "eng"))
# Expected: Hello, how are you?

# Tamil → Hindi
print(translate_onemt("இன்று வானிலை நன்றாக உள்ளது.", "tam", "hin"))
# Expected: आज मौसम अच्छा है।

# Batch translation
sentences = ["Good morning.", "Thank you.", "See you tomorrow."]
results   = translate_batch(sentences, sl="eng", tl="hin", batch_size=32)
for src, tgt in zip(sentences, results):
    print(f"{src}  →  {tgt}")
```

> ⚠️ **Important:** The current `model.safetensors` file has a broken fairseq→HF conversion. The model loads successfully but produces garbled output. You need to obtain properly converted weights for this to work. See the [Model Weights](#model-weights) section below.

---

## How Tokenization Works

Unlike standard MBart, OneNMT v3b uses a two-step tokenization pipeline:

```
source text
  → prepend language tag:   ###eng_Latn-to-tel_Telu### {text}
  → SentencePiece encoding  →  subword pieces
  → fairseq dict lookup     →  integer IDs   ← NOT SPM-native IDs
  → MBart encoder
```

The fairseq dictionary orders tokens by **training frequency** (descending), not SPM's internal ordering. This means the same subword piece maps to completely different IDs in the two systems — for example, `T` is ID 29 in the fairseq dict but ID 35376 via the HF tokenizer. Wrong IDs → wrong embeddings → garbled output.

The `hf_inference.py` script handles this correctly using `onemtv3b_spm.model` for segmentation and `fairseq_dict.json` for ID lookup.

---

## Repository Files

| File | Description |
|---|---|
| `model.safetensors` | Model weights  |
| `config.json` | HF model architecture config |
| `generation_config.json` | Default generation parameters |
| `fairseq_dict.json` | **Required** — fairseq vocab ID mapping (piece → ID) |
| `onemtv3b_spm.model` | **Required** — SentencePiece model for subword segmentation |
| `hf_inference.py` | Ready-to-use inference script |

> `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` are present for HF Hub format compliance but **must not be used for inference** — they produce incorrect token IDs due to the SPM vs fairseq ordering mismatch described above.

---

## Generation Parameters

| Parameter | Value | Notes |
|---|---|---|
| `decoder_start_token_id` | 2 | EOS token — fairseq convention (not BOS) |
| `forced_bos_token_id` | None | No forced BOS |
| `num_beams` | 5 | Beam search |
| `max_new_tokens` | 256 | Matches model's trained max position |
| `no_repeat_ngram_size` | 3 | Reduces repetition |
| `repetition_penalty` | 1.3 | Additional repetition control |

---

## Model Weights

### Current Status
The repository includes `model.safetensors` (4.4GB), but **it has a broken fairseq→HuggingFace conversion**. The model loads successfully (1.1B parameters) but produces garbled output because the weight mapping between fairseq and HF MBart architectures is incorrect.

### What's Fixed
- ✅ Tokenization pipeline (`hf_inference.py`) - correctly uses SPM + fairseq dict
- ✅ Config parameters - added missing normalization flags
- ✅ Input guards - enforces 256 position limit
- ✅ Generation parameters - beam search, repetition control

### Making It Your Own (Recommended Approach)
Instead of trying to fix the broken weights, **fine-tune a working base model** on your own data. This creates a genuinely new, unique model that's yours.

#### Step 1: Download a Working Base Model
```bash
# Use the provided script to download facebook/mbart-large-50-one-to-many-mmt
python fine_tune_example.py  # Edit the script first to add your data
```

Or download manually:
```bash
huggingface-cli download facebook/mbart-large-50-one-to-many-mmt
```

#### Step 2: Prepare Your Training Data
Create a CSV or JSON file with translation pairs:
```csv
source,target
"Hello, how are you?","नमस्ते, आप कैसे हैं?"
"Good morning","सुप्रभात"
"Thank you","धन्यवाद"
```

#### Step 3: Fine-Tune
Use the provided `fine_tune_example.py` script as a starting point. You'll need:
- At least 1,000-10,000 translation pairs for good results
- A GPU with 8GB+ VRAM (or use Google Colab)
- 1-3 hours of training time

#### Step 4: Replace Model Weights
After fine-tuning, replace `model.safetensors` with your new `my_finetuned_model/pytorch_model.bin` and update `config.json` accordingly.

### Verification
Once you have working weights, test with:
```bash
python -c "from hf_inference import translate_onemt; print(translate_onemt('Hello, how are you?', 'eng', 'hin'))"
```

Expected output: `नमस्ते, आप कैसे हैं?` (or similar correct Hindi translation)


## License

MIT