---
name: run-tests
description: Use when running the PySymGym test suite. Thin pointer to docs/testing.rst, the single source of truth for the taxonomy, commands, and fixtures; plus the machine-specific pitfalls (Poetry environment, root runner).
---

# Run tests

What the testing system is — the ``unit``/``integration``/``e2e`` taxonomy, the
exact commands, the fixtures, and where to add tests — is documented in
``docs/testing.rst``. Read that page before running anything. It is the single
source of truth; do not restate it here.

## Quick reference

- One canonical command from the repository root: ``poetry run pytest`` runs the
  fast unit tier. The same tiers are wrapped as ``make test-unit`` /
  ``make test-integration`` / ``make test-all`` / ``make test-cov``.
- CI selects markers per component explicitly (``-m "not e2e"`` for AIAgent and
  compstrat, ``-m e2e`` for runstrat); the root default filter deselects
  ``integration`` and ``e2e``.

## Do not run bare pytest

Always run pytest through Poetry (``poetry run pytest`` or ``make ...``), never
with the system interpreter. The system Python does not have the project's
dependencies (``torch``, ``torch_geometric``, ``mlflow``, ...) installed; a bare
``pytest`` silently resolves to a different interpreter and produces import
errors or stale results.

## Re-sync the environment after `poetry add` / `poetry remove`

``poetry add <package>`` / ``poetry remove <package>`` rewrite ``pyproject.toml``
and ``poetry.lock`` but do not install the resulting environment. After either,
restore it before running anything:

    poetry install
