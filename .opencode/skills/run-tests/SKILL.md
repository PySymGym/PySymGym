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

## Notes

- The runstrat suite needs two pre-built .NET artifacts first — see the
  "Per-component build prerequisites" section of `docs/setup.rst`.

## Local e2e reproduction (quality gate)

The e2e training pipeline (`.github/workflows/e2e_build_and_run.yml`) is
reproduced locally as part of the quality gate; the workflow file is the
source of truth for the steps. Machine-specific pitfalls:

- A stale `DOTNET_ROLL_FORWARD=Major` in the ambient environment rolls the
  net8.0 game server forward to a newer runtime, which changes the
  path-condition encoding and breaks e2e validation (the dataset is generated
  with the .NET 8 runtime; CI runs .NET 8). The broker now pins the runtime
  at launch (`DOTNET_ROLL_FORWARD=LatestPatch`, #580), so the broker-launched
  servers are immune; the dataset-generation step is a plain `dotnet`
  invocation, so run it under `env -u DOTNET_ROLL_FORWARD bash -lc '...'`
  and sanity-check `which dotnet` → `/home/gsv/.dotnet/dotnet` (the install
  with the .NET 8 runtime the dataset was generated with; `dotnet --version`
  may report a newer SDK — the runtime list is what matters). The same
  resolution applies to the whole Python phase, not just the generator: the
  configs' `launch_command` is a bare `dotnet` that the broker's and the
  svms_model validation's server children inherit via PATH, so put
  `/home/gsv/.dotnet` first on PATH for the training/broker steps too —
  otherwise the servers die at start with "framework 8.0.0 not found"
  (the system dotnet at `/usr/lib/dotnet` has no .NET 8 runtime).
- Start from a clean `AIAgent/report/` and `AIAgent/mlartifacts/` (CI runs
  from a fresh checkout): a stale `report/dataset_path_condition` confounds
  the tuning run. Also clean the generator's output directory
  (`GameServers/VSharp/VSharp.ML.GameServer.Runner/bin/Release/net8.0/8100`)
  before each generation: a leftover `SerializedEpisodes` from an interrupted
  run confounds the next one.
