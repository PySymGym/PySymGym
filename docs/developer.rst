Developer Guide
===============

Contribution guidelines
-----------------------

**Branches.** Work on a feature branch created from ``main``; never commit
directly to ``main``. Integration happens exclusively through a pull request
targeting ``main``, merged with rebase and merge (fast-forward) so ``main``
stays linear and the per-subtask commit structure is preserved.

**Commit messages.** Conventional Commits with exactly one subtask identifier::

    <type>(<task-issue>-S<n>): <summary>

where ``<type>`` is one of ``feat``, ``fix``, ``refactor``, ``docs``, ``test``,
``chore``. Ranges, lists, or comma-separated identifiers are forbidden: one
commit per completed atomic subtask. The body must explain why the change was
required.

**Issue closing.** The last subtask's commit carries the closing keyword for
the task's own issue (``Closes #<N>``) plus one per linked issue it fully
resolves (``Fixes #N`` for defects, ``Closes #N`` otherwise), each on its own
standalone line. A partially addressed linked issue uses a bare ``#N``.

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

CI as source of truth
---------------------

The CI workflows are the source of truth for the exact commands they run. If a
command in this guide and a workflow disagree, the workflow wins; fix this
guide.
