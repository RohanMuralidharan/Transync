# Architecture

Transync separates inference into small modules:

- `config.py` resolves runtime configuration and artifact paths.
- `languages.py` maps user-facing language codes to FLORES-style tags.
- `tokenizer.py` applies SentencePiece segmentation and fairseq dictionary ID
  lookup.
- `translator.py` loads the model and performs generation.
- `inference.py` exposes cached convenience wrappers.
- `cli.py` provides command-line access.

## Inference Flow

```text
text + language codes
  -> resolve source and target FLORES tags
  -> prepend the model language-pair tag
  -> segment with tokenizer/transync_spm.model
  -> map pieces through tokenizer/fairseq_dict.json
  -> generate with models/transync/model.safetensors
  -> map target IDs back to pieces
  -> SentencePiece decode
```

The fairseq dictionary mapping is part of the model contract. It is not an
implementation detail that can be swapped out without changing behavior.
