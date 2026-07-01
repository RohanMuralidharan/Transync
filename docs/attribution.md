# Attribution

Transync packages model artifacts, tokenizer assets, and inference behavior for
Indic multilingual neural machine translation.

## Project Identity

- Project name: Transync
- Repository: `https://github.com/RohanMuralidharan/Transync`
- Model family: Transync multilingual NMT
- License: MIT for Transync source code

## What Transync Preserves

- The `MBartForConditionalGeneration` model wrapper.
- The custom SentencePiece model.
- The fairseq source and target dictionary ID mapping.
- The language tag format: `###src_flores-to-tgt_flores### {text}`.
- EOS as `decoder_start_token_id`.
- Default generation values from the original inference script.

## Required Tokenizer Notice

Do not replace the tokenizer with `AutoTokenizer` or a standard MBart tokenizer
for inference with these artifacts. The SentencePiece IDs do not match the
fairseq dictionary IDs used during training.

## Citation

```bibtex
@software{transync2026,
      title={Transync: Indic Multilingual Neural Machine Translation},
      author={Transync Contributors},
      year={2026},
      url={https://github.com/RohanMuralidharan/Transync},
}
```
