Developer Guide
===============

*Contribution model, tests, style checks, docs build, and CI.*

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

**Quality gate.** No pull request is opened until the quality gate passes. See
the `quality-gates` skill and the sections below.

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

CI as source of truth
---------------------

The CI workflows are the source of truth for the exact commands they run. If a
command in this guide and a workflow disagree, the workflow wins; fix this
guide.
