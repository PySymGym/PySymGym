:description: Contribution model, tests, style checks, docs build, and CI.
:group: Development

Developer Guide
===============

Contribution guidelines
-----------------------

**Branches.** All work targets an *integration branch*. By default this is
``main``; a developer redirects it to a personal, long-lived branch (for
example ``gsv``) with a local, uncommitted git config key:

.. code-block:: console

    git config pysymgym.integrationBranch gsv

When the key is unset the integration branch is ``main``. Each task is
developed on its own feature branch created from the integration branch; never
commit directly to the integration branch. Integration always uses rebase and
merge (fast-forward) so history stays linear and the per-subtask commit
structure is preserved. There are two modes:

- **Direct mode** (integration branch is ``main``): each task is integrated by
  a pull request targeting ``main``.
- **Stacked mode** (integration branch is a personal branch): each task is
  integrated by rebasing its feature branch onto the integration branch and
  fast-forward merging it; **no pull request is opened**. Several tasks thus
  accumulate on the integration branch. A single pull request from the
  integration branch to ``main`` is opened only on the user's explicit
  request.

A personal integration branch is long-lived: it is never deleted and is
periodically rebased onto ``main`` (at task boundaries and immediately before
the final pull request) to limit divergence. Before the final pull request,
the whole ``main...integration`` diff passes an aggregated code review and the
quality gate.

**Commit messages.** Conventional Commits with exactly one subtask identifier::

    <type>(<task-issue>-S<n>): <summary>

where ``<type>`` is one of ``feat``, ``fix``, ``refactor``, ``docs``, ``test``,
``chore``. Ranges, lists, or comma-separated identifiers are forbidden: one
commit per completed atomic subtask. The body must explain why the change was
required.

**Issue closing.** The last subtask's commit carries the closing keyword for
the task's own issue (``Closes #<N>``) plus one per linked issue it fully
resolves (``Fixes #N`` for defects, ``Closes #N`` otherwise), each on its own
standalone line. A partially addressed linked issue uses a bare ``#N``. The
closing keyword takes effect when the commit reaches ``main``: in stacked mode
the issue stays open until the final pull request to ``main`` is merged.

A task is done when its subtask commits are on the integration branch, even if
the linked issue has not yet closed.

**Task tracking.** Each task is a GitHub issue labeled ``task``. A set of
related tasks is grouped under a **hub** issue labeled ``hub``: the task issues
are attached to the hub as GitHub sub-issues (through the REST API, not a link
in the body), the batch's global plan is mirrored onto the hub as a comment, and
its progress is updated as tasks integrate. The hub closes with the last task of
the batch (``Closes #<hub>`` in that task's final commit). The operational
procedure lives in the ``workflow-management`` and ``planning`` skills.

**Quality gate.** No pull request is opened until the quality gate passes. See
the `quality-gates` skill and the sections below.

**Pull requests.** Every change updates its documentation: user-visible
behavior is documented under ``docs/`` (with the ``:description:`` and
``:group:`` metadata the navigation map is generated from), and ``README.md``
links must resolve. The pull request template lists these checks, and the docs
build (``.github/workflows/docs.yaml``) enforces them. See ``CONTRIBUTING.md``
for the pointers.

Test pipeline
-------------

Tests are run per component with pytest:

.. code-block:: console

    poetry run pytest tests -sv

from ``AIAgent/``, ``tools/compstrat/``, and ``tools/runstrat/``. CI runs the
same commands in the "Run main repo tests" and "Run compstrat tool tests"
steps of ``.github/workflows/python_tests.yaml``.

There is no coverage threshold; the suite is expected to report 0 failures and
0 skipped.

Quality checks
--------------

Linting and formatting use `ruff <https://docs.astral.sh/ruff/>`_:

.. code-block:: console

    poetry run ruff check
    poetry run ruff format --check

The ruff configuration (target version, enabled rule sets, included paths)
lives in ``pyproject.toml``. CI runs the same two commands in
``.github/workflows/python_linting.yaml``.

The `ruff editor integrations <https://docs.astral.sh/ruff/integrations/>`_
(for example the `VSCode
<https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`__
extension) run the same checks on save.

Docs build
----------

The Sphinx sources live in ``docs/``. Build them locally with:

.. code-block:: console

    poetry install --only docs
    poetry run sphinx-build -W --keep-going -b html docs docs/_build/html

The build runs under the no-warnings policy (``-W --keep-going``): any warning
fails the build. CI runs the same command in ``.github/workflows/docs.yaml``.

Published docs
--------------

The built HTML is published to GitHub Pages at
`pysymgym.github.io/PySymGym <https://pysymgym.github.io/PySymGym/>`_. The
``deploy`` job in ``.github/workflows/docs.yaml`` runs only for pushes to
``main``; pull requests build without deploying. Read the Docs is planned as a
secondary mirror.

Runner requirements
-------------------

Four workflow files contain jobs that run on a self-hosted runner
(``jbLabSelfHostedCI``) rather than on GitHub-hosted compute:
``e2e_build_and_run.yml`` (the shared pipeline called by
``build_and_run.yaml`` and ``build_and_run_model_val.yaml``),
``build_and_test_usvm.yaml``, ``publish_image.yaml`` and
``runstrat_tool.yaml``. All of those jobs also declare a ``container:``, so
the work happens inside a container while the workflow itself is still driven
by the runner.

Every action referenced by ``.github/workflows/`` runs on Node.js 24, which
requires Actions Runner ``v2.327.1`` or newer. The GitHub-hosted runners
satisfy this by construction, but the self-hosted runner does not: on one older
than ``v2.327.1`` every action in those four workflow files fails to start,
taking the whole training, image-publishing and tool-testing pipeline down with
it.
Keep that runner current before bumping an action to a release that raises its
runtime requirement.

The runner's installed version is not visible from the repository, so this is
the one CI constraint that cannot be checked by reading the workflow files.

Self-hosted end-to-end workflows
--------------------------------

The policies below keep the long self-hosted pipelines deterministic and stop
them from piling up on the single runner. This section explains the
non-obvious decisions behind the pipelines; the workflow files are the source
of truth for the exact steps.

**The two e2e workflows share one pipeline.** ``build_and_run.yaml`` and
``build_and_run_model_val.yaml`` differ only in their name, concurrency group,
and training inputs, so their common pipeline (Dockerfile hash, checkout,
toolchain setup, V# and maps build, data generation, training with MLflow,
artifact upload, sanity check) lives once in the reusable workflow
``.github/workflows/e2e_build_and_run.yml`` (``workflow_call`` only — it never
runs on its own). Each e2e workflow file is a thin caller that selects the
pipeline mode through two inputs: ``training-config`` (the config of the main
training run, relative to ``AIAgent/``) and the optional
``improvement-base-config`` (see below). To add another e2e variant, create
such a thin caller with its own name and concurrency group — do not copy the
pipeline.

**Servers live in the step that uses them.** MLflow and the game-server
broker are started inside the same step as the training command that talks to
them, gated on readiness polling (an HTTP ``/health`` check for MLflow, a TCP
connect to the broker port) instead of fixed sleeps, and cleaned up when the
step exits. A background process does not survive a step boundary, so a
server started in an earlier step is gone by the time a later step needs it;
a fixed sleep additionally races with the variable startup time.

**Triggers and concurrency.** The four self-hosted workflows that build or
test code (``build_and_run.yaml``, ``build_and_run_model_val.yaml``,
``runstrat_tool.yaml``, ``build_and_test_usvm.yaml``) run only on pushes to
``main`` and on pull requests — feature-branch pushes do not trigger them, so
a dependabot bump does not trigger the long e2e pipeline twice (push and pull
request) and stale runs do not queue up. The two e2e workflows additionally
use a per-workflow concurrency group keyed by ref that cancels in-progress
runs for pull requests only: a new push to the same branch supersedes the
stale run, while a run started by a push to ``main`` is never cancelled.

**Dataset improvement continues from tuning.** In ``build_and_run.yaml`` the
tuning run and the dataset-improvement run share one MLflow experiment. The
improvement is an optional mode of the shared pipeline, selected by that
workflow's ``improvement-base-config`` input: after the main (tuning) run,
``derive_dataset_improvement_config.py`` queries the same-step server for the
best trial's ``model.pth`` and ``trial.pkl`` artifact URIs and writes them
into a copy of the base config that the improvement step consumes. The base
config remains the single source of truth — the pipeline derives the URIs from
the tuning run instead of hard-coding them. ``build_and_run_model_val.yaml``
leaves the input empty and runs only the main training run.

**Unexhausted steps fail CI.** Both e2e validation workflows run with
``fail_on_unexhausted_steps`` enabled (flag semantics in
:doc:`usage`, "SVM validation failure flags"). The symbolic engine ending a
map before all planned steps are played without 100% coverage is an engine
defect; in CI it must fail the run instead of being swallowed by a warning,
while local runs keep the default warning-only behavior.

**Host GPU spike (#567).** Issue #567 asked whether the end-to-end training
should run on the self-hosted runner's host GPU (reported: NVIDIA GT 1030,
Pascal, ~2 GB VRAM). Dedicated self-hosted e2e runs were off the table for
this batch, so the spike was measured locally on a dev machine with an NVIDIA
GeForce MX150 (GP108M — the same Pascal chip class and 2 GB VRAM as the GT
1030), which makes the measurement representative of the runner.

Two findings frame the measurement:

- The CI environment already ships a CUDA-enabled torch. The Docker image
  (``.github/docker/Dockerfile``) provides only ubuntu and the dotnet SDKs;
  Python dependencies are installed at runtime by ``poetry install`` from the
  lock, and the locked torch resolves on Linux x86_64 to the manylinux wheel
  with all ``nvidia-*-cu12`` dependencies (cuDNN, cuBLAS, NCCL, ...). So
  "provide a CUDA-enabled torch build inside the CI image" required no image
  change; the only missing piece for GPU use is device passthrough to the job
  container.
- Device selection needs no code change either: ``AIAgent/config.py`` picks
  ``cuda:0`` when available and falls back to CPU, so a run without GPU
  passthrough is exactly what CI does today (CUDA-capable wheel, no device).

The measurement replicates the shared e2e pipeline step for step (V# server
and maps build, data generation, MLflow + game-server broker with readiness
polling, ``run_training.py --config ../workflow/config_for_tests.yml`` from
``AIAgent/``) in the locked environment (torch 2.7.1+cu126). The CPU baseline
hides the GPU with ``CUDA_VISIBLE_DEVICES=""``; the GPU run leaves it
visible. Wall times below cover the training step — the phase a GPU could
affect; builds and data generation are device-independent.

CPU baseline (two runs):

.. list-table::
   :header-rows: 1
   :widths: 15 30 25

   * - run
     - training step (s)
     - peak RSS (MiB)
   * - 1
     - 130.6
     - 742
   * - 2
     - 131.7
     - 742

GPU run (one run, device ``cuda:0``):

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - training step (s)
     - peak RSS (MiB)
     - peak VRAM (MiB)
   * - 134.0
     - 863
     - 296 total (~72 torch; 224 is the desktop compositor baseline)

The workload fits in the 2 GB card with a wide margin (~1.8 GB free after
the compositor), so "does it fit" is not the constraint. The question is
speed, and the answer is no: the GPU run (134.0 s) is not faster than the
CPU baseline (130.6 / 131.7 s) — a difference within run-to-run noise.

Why the GPU does not help here: the training step is dominated by the
``svms_each_step`` validation (~54–56 s of the ~132 s): a .NET game server
plays each of the 35 maps for 200–500 steps, and at every step the model
picks the next action — one small graph forward pass with a CPU<->GPU round
trip per step (see ``ml/predict.py``). The training epochs themselves are
no-ops in this workload: the dataset only keeps maps that reached 100%
coverage (``threshold_coverage: 100``), which a fresh random model never
does. So the entire torch-bound part is per-step micro-batch inference,
where the transfer and kernel-launch overhead cancels the GP108's compute
advantage — not faster on the GPU.

**Decision: drop.** Running the e2e training on the host GPU is not worth
it: no measured speedup (slightly slower within noise), the torch-bound part
is a minority of the wall time, and enabling it would add a host
prerequisite and passthrough fragility for zero benefit. The CI image keeps
the CUDA-enabled torch it already ships (harmless without a device); no
workflow change is made.

No code change accompanies this decision: the shared e2e pipeline gains no
GPU passthrough, and the Dockerfile is untouched. Wiring ``--gpus all`` into
the build-and-launch container would add a host prerequisite
(nvidia-container-toolkit) and a new failure mode to a long self-hosted
pipeline for zero measured benefit.

If this is ever revisited, the opt-in point is the ``container.options`` of
the build-and-launch job in ``.github/workflows/e2e_build_and_run.yml``
(``--gpus all``, default off so CPU runs are unaffected). The passthrough
flag itself can be host-dependent: on the dev machine used for this
measurement, docker rejects CDI mode (``--gpus all``) and requires
``--runtime nvidia`` instead, so a revisit must check the runner host's
docker configuration first. Re-measure before wiring if a future workload
makes the torch-bound part dominant in wall time (larger models or datasets),
or if the runner's GPU class changes.

CI as source of truth
---------------------

The CI workflows are the source of truth for the exact commands they run. If a
command in this guide and a workflow disagree, the workflow wins; fix this
guide.
