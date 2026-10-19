---
name: quality-gates
description: Use before integrating a task. Defines the hard gate that must pass (tests + style + docs build + full local CI reproduction) and how to interpret its result. References the CI workflows for the exact commands.
---

# Quality Gates

The hard gate a task must pass before integration. It has exactly two terminal
states: **PASS** or **BLOCKED**. There is no "pass with exceptions". In stacked
mode the same gate is also run over the whole `main...integration` diff
immediately before the final pull request to `main`.

## What the gate is

The gate is the combination of the test suite, the style/lint checks, the
docs build, and a full local reproduction of CI:

- **Tests** — see `.github/workflows/python_tests.yaml` for the exact command
  and working directories (see the `run-tests` skill). 0 failures, 0 skipped.
- **Style/lint** — see `.github/workflows/python_linting.yaml` for the exact
  commands (`ruff check`, `ruff format --check`; see the `code-style` skill).
- **Docs build** — see `.github/workflows/docs.yaml`. Sphinx builds under the
  no-warnings policy (`-W --keep-going`): any warning fails the build, so the
  exit code is sufficient.
- **Full CI reproduction** — at the end of the task, run every step of every
  PR-triggered workflow locally (each `.github/workflows/*.y*ml` is the source
  of truth for its own steps). The e2e training pipeline
  (`e2e_build_and_run.yml`, called by `build_and_run.yaml` and
  `build_and_run_model_val.yaml`) is mandatory: it is the only step that runs
  the model on a GPU, while `python_tests` runs on CPU-only runners —
  device-dependent code paths (CUDA) are invisible to the unit suite.

The CI workflows are the source of truth for the commands they run; this skill
only defines the gate semantics. There is no type-check step and no coverage
threshold in this project.

## Procedure

1. Run the tests (`poetry run pytest tests -sv`) from every component the
   change touches (`AIAgent/`, `tools/compstrat/`, `tools/runstrat/`). 0
   failures, 0 skipped.
2. Run `ruff check` and `ruff format --check` (see
   `.github/workflows/python_linting.yaml`). No errors.
3. Build the docs (see `.github/workflows/docs.yaml`). It must exit 0
   (no-warnings policy).
4. Run all CI steps locally (see "Full CI reproduction" above), including the
   e2e training pipeline on a GPU when one is available.
5. Interpret the result:
    - All clean → **PASS**. Proceed to merge (see `git-workflow`).
    - Any failure → **BLOCKED**.
    - A CI step that cannot be run locally (missing toolchain, no GPU) is not
      a pass: report it as **BLOCKED** with the reason — never silently skip.

## On BLOCKED

- STOP. Do not merge. Do not mark the task done.
- Do **not** assess whether a failure is pre-existing or unrelated to your
  changes — fix it regardless.
- Do **not** weaken, skip, or comment out failing tests to make the suite green
  (see the Blocked Work Protocol in the `subtask-loop` skill).
- Fix every failure and re-run until **PASS**.
