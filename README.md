# Transync
Transync is a multilingual transformer-based translation model for accurate, scalable, and offline machine translation across multiple languages.
---
language:
  - eng
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

license: mit

pipeline_tag: translation

tags:
  - translation
  - multilingual
  - transformer
  - seq2seq
  - machine-translation
  - offline
---

# 🌍 Transync

**Transync** is a multilingual Neural Machine Translation (NMT) model built for fast, accurate, and scalable translation across English and multiple Indian languages. It is designed for offline deployment, multilingual AI applications, document translation, and research.

---

## ✨ Features

- 🌍 Multilingual Translation
- ⚡ Offline Inference
- 🔄 English ↔ Indian Languages
- 🌐 Indian Language ↔ Indian Language
- 📦 Hugging Face Compatible
- 🚀 High-performance Transformer Architecture

---

## 📊 Model Overview

| Property | Value |
|-----------|-------|
| Model Type | Neural Machine Translation |
| Architecture | Transformer Encoder–Decoder |
| Parameters | ~1.1B |
| Framework | PyTorch |
| Inference | Hugging Face |

---

## 🌐 Supported Languages

- English
- Hindi
- Telugu
- Tamil
- Kannada
- Malayalam
- Marathi
- Bengali
- Gujarati
- Punjabi
- Urdu
- Assamese
- Nepali
- Odia
- Sanskrit
- Maithili
- Bodo
- Dogri
- Konkani
- Meitei
- Santali
- Kashmiri
- Sindhi

---

# 🚀 Installation

```bash
pip install -r requirements.txt
```

---

# 💻 Example

```python
from hf_inference import translate_transync

print(
    translate_transync(
        "Hello, how are you?",
        "eng",
        "hin"
    )
)
```

---

# 📂 Repository

```
Transync/
│
├── model.safetensors
├── config.json
├── generation_config.json
├── hf_inference.py
├── fairseq_dict.json
├── transync_spm.model
├── requirements.txt
└── README.md
```

---

## 📄 Applications

- Machine Translation
- Offline Translation
- PDF Translation
- Document Translation
- Multilingual AI
- Research
- Education

---

## 📜 License

MIT License

---

# 👨‍💻 Developer

**Rohan M**

AI Engineer • Machine Learning • Open Source

---

## ⭐ Support

If you find **Transync** useful, consider starring the repository and following future updates.
````

This gives you a clean, professional **Transync** brand while keeping the technical details truthful.

If, later, you actually replace `onemtv3b_spm.model`, create your own vocabulary, or change the architecture, then you can rename those components too.
