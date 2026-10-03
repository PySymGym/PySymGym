---
name: reusing
description: Use before creating any new code, types, functions, or documentation. Run this reuse checklist to find existing material to reuse or generalize instead of duplicating. Enforces the "one source of truth / no duplicates" principle.
---

# Reusing

Before writing new code or docs, search for existing material and reuse or
generalize it. Duplicating an existing function, type, or doc section violates
the project's single-source-of-truth principle.

## Checklist

Run through these before creating new material:

- [ ] **Agent code** — search `AIAgent/` for an existing function, class, or
      type that already does (or nearly does) the job. Prefer generalizing the
      existing one over adding a near-copy. Start from `AIAgent/common/` for
      shared data types and utilities.
- [ ] **Config** — settings flow through `AIAgent/config.py` and the YAML files
      in `configs/`; extend the existing config structures rather than adding a
      parallel one.
- [ ] **Servers/connection** — the broker and game-server protocol lives in
      `AIAgent/connection/`; reuse those clients and message types instead of
      re-implementing transport.
- [ ] **ML** — models and training utilities live in `AIAgent/ml/` (`models/`,
      `training/`, `validation/`); reuse the dataset, protocol, and
      torch-geometric model abstractions before writing new ones.
- [ ] **Tools** — check `tools/compstrat`, `tools/runstrat`,
      `tools/dataset_tools`, and `tools/util` for an existing CLI/helper before
      adding a new one.
- [ ] **Tests** — check the component's `tests/` directory (`AIAgent/tests`,
      `tools/*/tests`) for existing fixtures, helpers, or patterns to extend
      rather than duplicate.
- [ ] **Docs** — check the navigation map (`docs/index.rst`) and
      `docs/architecture.rst`, plus existing docstrings, for a section to
      update rather than add a new page. New pages need `:description:` and
      `:group:` metadata.

## Rules

- If a close match exists, **generalize it** (make it generic enough to cover
  both cases) rather than copy-paste.
- Record what is reused in each subtask's **Code** section of
  `tasks/detailed_plan.md`.
- When in doubt, prefer the existing abstraction; only create new material when
  no reasonable generalization is possible.
