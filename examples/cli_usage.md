# CLI Usage

Install Transync in editable mode:

```bash
pip install -e .
```

Translate a single sentence:

```bash
transync --source eng --target hin --text "Hello, how are you?"
```

Translate one segment per line from a file:

```bash
transync --source eng --target tel --input-file examples/input.txt --output-file output.txt
```

Run the CLI directly from a source checkout:

```bash
python scripts/translate.py --source eng --target hin --text "Good morning."
```
