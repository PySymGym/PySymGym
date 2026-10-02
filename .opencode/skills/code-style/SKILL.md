---
name: code-style
description: Use before committing to format and lint PySymGym. Thin pointer to the "Linting tools" section of README.md and the CI workflow step in .github/workflows/python_linting.yaml, which hold the exact commands.
---

# Code style

What the quality checks enforce (ruff lint and format) and the exact commands
are documented in the "Linting tools" section of `README.md`; CI runs the same
checks (see `.github/workflows/python_linting.yaml`). Read that section before
running anything.

## Notes

- The linter/formatter is `ruff`; it lives in the `formatter` dependency group
  in `pyproject.toml`, so run it via `poetry run`.
- `pyproject.toml` holds the ruff configuration (`[tool.ruff]`), including the
  files linted (`AIAgent/**/*.py`, `tools/**/*.py`, notebooks).
