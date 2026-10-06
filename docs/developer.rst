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

Five workflows run on a self-hosted runner (``jbLabSelfHostedCI``) rather than
on GitHub-hosted compute: ``build_and_run.yaml``,
``build_and_run_model_val.yaml``, ``build_and_test_usvm.yaml``,
``publish_image.yaml`` and ``runstrat_tool.yaml``. Most of their jobs also
declare a ``container:``, so the work happens inside a container while the
workflow itself is still driven by the runner.

Every action referenced by ``.github/workflows/`` runs on Node.js 24, which
requires Actions Runner ``v2.327.1`` or newer. The GitHub-hosted runners
satisfy this by construction, but the self-hosted runner does not: on one older
than ``v2.327.1`` every action in those five workflows fails to start, taking
the whole training, image-publishing and tool-testing pipeline down with it.
Keep that runner current before bumping an action to a release that raises its
runtime requirement.

The runner's installed version is not visible from the repository, so this is
the one CI constraint that cannot be checked by reading the workflow files.

Self-hosted end-to-end workflows
--------------------------------

Three policies keep the long self-hosted pipelines deterministic and stop them
from piling up on the single runner. The workflow files are the source of
truth for the exact steps; this section only explains the non-obvious
decisions behind them.

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
tuning run and the dataset-improvement run share one MLflow experiment; after
tuning, ``derive_dataset_improvement_config.py`` queries the same-step server
for the best trial's ``model.pth`` and ``trial.pkl`` artifact URIs and writes
them into a copy of the base config that the improvement step consumes. The
base config remains the single source of truth — the workflow derives the
URIs from the tuning run instead of hard-coding them.

**Unexhausted steps fail CI.** Both e2e validation workflows run with
``fail_on_unexhausted_steps`` enabled (flag semantics in
:doc:`usage`, "SVM validation failure flags"). The symbolic engine ending a
map before all planned steps are played without 100% coverage is an engine
defect; in CI it must fail the run instead of being swallowed by a warning,
while local runs keep the default warning-only behavior.

CI as source of truth
---------------------

The CI workflows are the source of truth for the exact commands they run. If a
command in this guide and a workflow disagree, the workflow wins; fix this
guide.
