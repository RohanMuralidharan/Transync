# Contributing

Thanks for helping improve Transync.

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Contribution Guidelines

- Keep model inference behavior unchanged unless the pull request explicitly
  explains why a behavioral change is required.
- Preserve applicable license notices for any third-party code, data, or model
  artifacts.
- Add or update tests for code changes where practical.
- Prefer `pathlib`, type hints, clear exception messages, and standard logging.
- Do not commit generated caches, local outputs, or temporary files.

## Checks

```bash
python -m compileall src tests examples scripts
pytest
```

Large model files should be committed with Git LFS.
