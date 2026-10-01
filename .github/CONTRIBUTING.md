# Contributing to ObsidianRAG

Thanks for taking the time to contribute! This project aims to stay small,
fully offline by default, and easy to reason about.

## Ways to contribute

- **Report bugs:** open an issue with a clear title and a minimal reproduction.
- **Suggest features:** open an issue describing the problem and your idea.
- **Send a pull request:** fork the repo, branch from `main`, and open a PR.

## Development setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/AkashNaickar/ObsidianRAG.git
cd ObsidianRAG
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before you open a pull request

Run the same checks CI runs:

```bash
ruff check .
ruff format --check .
pytest
```

- `pytest` enforces a coverage floor; add tests for new behaviour.
- Keep the default pipeline dependency-light: new runtime dependencies need a
  strong justification, and offline embedders must stay deterministic.

## Style

- Source code lives in `src/obsidian_rag/`; tests live in `tests/`.
- Follow PEP 8 (enforced by Ruff), keep line length to 100 characters.
- Write focused commits with imperative messages, e.g.
  `add overlap handling for long chunks`.

## Reporting security issues

Please follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
