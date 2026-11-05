---
name: run-tests
description: Use when running the PySymGym test suite. Thin pointer to docs/testing.rst, the single source of truth for the taxonomy, commands, and fixtures.
---

# Run tests

What the testing system is — the ``unit``/``integration``/``e2e`` taxonomy, the
exact commands, the fixtures, and where to add tests — is documented in
``docs/testing.rst``. Read that page before running anything. It is the single
source of truth; do not restate it here.

## Quick reference

- The fast unit tier runs across every component in one command:
  ``make test-unit`` (``poetry run pytest -m "not integration and not e2e"``).
  The same tiers are wrapped as ``make test-unit`` / ``make test-integration``
  / ``make test-all`` / ``make test-cov``.
- CI selects markers explicitly (``-m "not e2e"`` for the gated-tier ratchet
  step, ``-m e2e`` for runstrat); ``addopts`` carries no tier filter, so a
  node-ID invocation (``pytest path::test``) is never filtered by tier.

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

## Notes

- The runstrat suite needs two pre-built .NET artifacts first — see the
  "Per-component build prerequisites" section of `docs/setup.rst`.

## Local e2e reproduction (quality gate)

The e2e training pipeline (`.github/workflows/e2e_build_and_run.yml`) is
reproduced locally as part of the quality gate; the workflow file is the
source of truth for the steps. The .NET runtime requirement — the game
servers must run on the .NET 8 runtime the dataset was generated with, the
broker's `DOTNET_ROLL_FORWARD=LatestPatch` pin, and `env -u
DOTNET_ROLL_FORWARD` for the two non-broker invocations — is documented in
the "Local e2e runs" section of `docs/testing.rst`. Pitfalls:

- Make sure `dotnet` on PATH resolves to an install that includes the .NET 8
  runtime — check `dotnet --list-runtimes`, not `dotnet --version` (which may
  report a newer SDK). If the system dotnet lacks it, prepend that install's
  bin directory to PATH for the whole Python phase, not just the
  dataset-generation step: the configs' `launch_command` is a bare `dotnet`
  that the broker's and the svms_model validation's server children inherit
  via PATH, and a mismatched runtime either kills the servers at start with
  "framework 8.0.0 not found" or silently changes the path-condition
  encoding.
- Start from a clean `AIAgent/report/` and `AIAgent/mlartifacts/` (CI runs
  from a fresh checkout): a stale `report/dataset_path_condition` confounds
  the tuning run. Also clean the generator's output directory
  (`GameServers/VSharp/VSharp.ML.GameServer.Runner/bin/Release/net8.0/8100`)
  before each generation: a leftover `SerializedEpisodes` from an interrupted
  run confounds the next one.
