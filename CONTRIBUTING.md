# Contributing

Thanks for taking a look. This project aims to stay small, readable and dependable, so the bar for
new abstractions is high — a focused bug fix or test is worth more than a new subsystem.

## Getting started

```bash
git clone https://github.com/hamids-ai-automations/gemini-agent-automation-toolkit.git
cd gemini-agent-automation-toolkit
python -m venv .venv
.\.venv\Scripts\activate          # Windows
source .venv/bin/activate         # macOS / Linux
pip install -e ".[dev]"
pytest -q
```

## Before you open a pull request

1. **Keep tests green.** `pytest -q` must pass; the suite runs fully offline.
2. **Add tests for behaviour changes.** Anything touching approval, retries or redaction needs a
   regression test — those are the parts users trust us with.
3. **Type and lint.** `mypy gemini_agent_toolkit` and `ruff check .` should be clean.
4. **Document public API changes** in the docstring *and* `README.md` / `CHANGELOG.md`.
5. **Do not commit secrets** — no API keys, tokens, `.env` files or captured traffic.

## Style

- Python 3.10+, `from __future__ import annotations` at the top of every module.
- Public functions and classes need docstrings that state behaviour, not restate the signature.
- Prefer explicit exceptions from `gemini_agent_toolkit.exceptions` over bare `Exception`.
- No silent fallbacks: if an operation cannot be performed safely, raise.

## Commits and pull requests

- One logical change per PR; keep diffs reviewable.
- Commit messages: imperative mood, scoped where useful (`tools: validate int bounds`).
- Describe *why* in the PR body, plus how you verified the change.
- CI must be green; maintainers may ask for smaller commits or added tests.

## Reporting bugs

Include the toolkit version, Python version, a minimal reproduction, expected vs actual behaviour
and the relevant log lines **with secrets removed**. Security issues go to
[SECURITY.md](SECURITY.md), never a public issue.

## Feature requests

Open an issue describing the problem before the solution. If it is already on
[docs/ROADMAP.md](docs/ROADMAP.md), add your use case there so it can be prioritised.

By contributing you agree that your contributions are licensed under the project's MIT license and
that you will follow the [Code of Conduct](CODE_OF_CONDUCT.md).
