---
name: run-tests
description: Use when running the PySymGym test suite (pytest). Thin pointer to .github/workflows/python_tests.yaml, which holds the exact command and working directories, plus the machine-specific pitfalls (Poetry environment, per-tool test roots).
---

# Run tests

What the test pipeline is and where the exact commands live (the CI workflow
steps) are in `.github/workflows/python_tests.yaml`. Read that workflow before
running anything. PySymGym has no single top-level suite: tests are run per
component from its own directory.

## Commands

CI runs pytest from each component directory:

    poetry run pytest tests -sv

from `AIAgent/`, `tools/compstrat/`, and `tools/runstrat/`. Run the same
command from the component whose code you changed (or from all three when in
doubt).

## Do not run bare pytest

Always run pytest through Poetry (`poetry run pytest`), never with the system
interpreter. The system Python does not have the project's dependencies
(`torch`, `torch_geometric`, `mlflow`, ...) installed; a bare `pytest` silently
resolves to a different interpreter and produces import errors or stale
results.

## Re-sync the environment after `poetry add` / `poetry remove`

`poetry add <package>` / `poetry remove <package>` rewrite `pyproject.toml` and
`poetry.lock` but do not install the resulting environment. After either,
restore it before running anything:

    poetry install

## Notes

- Test dependency group: `test` in `pyproject.toml` (`pytest`).
- No coverage gate is configured; the suite is expected to report 0 failures
  and 0 skipped.
