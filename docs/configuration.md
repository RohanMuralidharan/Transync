# Configuration

The default configuration file is `configs/transync.json`.

```json
{
  "hf_repo": "RohanMuralidharan/Transync",
  "model_dir": "models/transync",
  "sentencepiece_model": "tokenizer/transync_spm.model",
  "dictionary_path": "tokenizer/fairseq_dict.json",
  "device": "auto",
  "long_text_word_threshold": 200,
  "chunk_word_size": 100,
  "generation": {
    "beam_size": 5,
    "max_new_tokens": 256,
    "early_stopping": true,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.3
  }
}
```

Use a custom config file with either:

```bash
TRANSYNC_CONFIG=/path/to/transync.json transync --source eng --target hin --text "Hello"
```

or:

```bash
transync --config /path/to/transync.json --source eng --target hin --text "Hello"
```

Set `"device"` to `"auto"`, `"cpu"`, `"cuda"`, or a specific PyTorch device such
as `"cuda:0"`.
