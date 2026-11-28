# LLMhubAudit — Agent Guidance

## Project overview

`llmhubaudit` is a Python 3.13 package (src layout) for auditing LLM hubs. It is intentionally a minimal shell: the package itself has no runtime dependencies and no business logic. Hub-specific audit scripts live under `scripts/` and are standalone (they do not import from `llmhubaudit`).

## Directory structure

```
LLMhubAudit/
├── AGENTS.md                 # this file
├── LICENSE
├── README.md
├── pyproject.toml            # package metadata + ruff/black/isort/mypy/pytest config
├── requirements.txt          # package runtime deps (none)
├── requirements-dev.txt      # dev tooling
├── requirements-test.txt     # test tooling
├── config.example.yaml       # example YAML config (copy to config.yaml)
├── .github/workflows/ci.yml  # GitHub Actions CI
├── .pre-commit-config.yaml   # pre-commit hooks
├── .gitignore
├── src/llmhubaudit/__init__.py   # package init (__version__ = "0.1.0")
├── tests/__init__.py         # tests package placeholder
└── src/llmhubaudit/ollama/   # Ollama audit package
```

## Key rules

- The `llmhubaudit.ollama` package (under `src/llmhubaudit/ollama/`) is the Ollama audit package.
- The `llmhubaudit` package has **no runtime dependencies** (`dependencies = []` in `pyproject.toml`). Hub-specific deps (ollama, requests, psutil, etc.) belong in the scripts that use them.
- Python minimum version: **3.13**.
- Ruff excludes `scripts/` from linting (`exclude = ["scripts"]` in `pyproject.toml`).

## Common commands

```bash
# Install package + dev tooling
pip install -e ".[dev]"

# Run ruff on the package source (scripts excluded by config)
ruff check src

# Run ruff format check
ruff format --check src

# Install pre-commit hooks
pre-commit install

# Run all pre-commit hooks
pre-commit run --all-files
```

## CI

`.github/workflows/ci.yml` runs on push/PR to `main`/`master`:
1. **lint** — ruff check, ruff format, black, isort, codespell
2. **type-check** — mypy --strict src
3. **security** — bandit -r src
4. **test** — pytest with coverage (currently no tests; placeholder)
5. **build** — python -m build + twine check (depends on all above)

## Adding a new hub script

1. Create `scripts/<hub>/<script>.py` as a standalone script.
2. Do not import from `llmhubaudit`.
3. Declare any third-party deps the script needs in the script's docstring or a hub-specific requirements file.
4. Ruff ignores `scripts/`, so the script won't fail lint hooks.

