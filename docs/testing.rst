:description: The local-first test system: taxonomy, commands, fixtures, and coverage.
:group: Development

Testing
=======

PySymGym uses a local-first pytest setup. ``poetry run pytest`` from the
repository root runs the fast **unit** tier across every component in one
command, without network, GPU, ``.NET``, or built binary fixtures. Slower tiers
are selected explicitly and the full end-to-end pipeline stays in CI.

This page is the single source of truth for the testing system; the developer
guide and the agent skills point here.

Test pyramid
------------

The suite is layered so most checks run fast and locally:

- **Unit** (the base): pure, deterministic tests of isolated logic. No network,
  GPU, subprocess, or binary fixtures. This is the default tier and must run in
  seconds.
- **Integration** (the middle): in-process tests that wire real components
  together with fakes and golden fixtures (for example the ONNX export, the
  pc-remover graph transform, and the compstrat pipeline).
- **E2E** (the top): the full pipelines that need built servers, maps and the
  ``.NET`` toolchain; they run in the existing self-hosted CI workflows.

A change should add the lowest tier that can catch its regression; the higher
tiers stay in CI where the required infrastructure is available.

Test taxonomy
-------------

Tests are classified with pytest markers:

``unit``
    Pure, deterministic tests with no network, GPU, subprocess, or binary
    fixtures.

``integration``
    In-process tests that use fakes and golden fixtures.

``e2e``
    Full pipeline tests that need built servers, maps, or the ``.NET``
    toolchain.

``gpu`` / ``network`` / ``slow`` / ``serial``
    Orthogonal constraints: require a CUDA device, require network, take
    noticeably long, or must not run in parallel.

The default ``addopts`` deselects ``integration`` and ``e2e``, so a plain
``poetry run pytest`` is the fast unit tier.

Commands
--------

.. code-block:: console

    poetry run pytest            # fast unit tier (default)
    poetry run pytest -m integration
    poetry run pytest -m e2e

The same tiers are wrapped in a root ``Makefile`` so nobody has to memorize the
invocations:

.. code-block:: console

    make test-unit           # fast unit tier (same as `poetry run pytest`)
    make test-integration    # -m integration
    make test-all            # everything (e2e needs the built toolchain)
    make test-cov            # unit tier with a coverage report

The configuration lives in ``[tool.pytest.ini_options]`` in the root
``pyproject.toml``.

To run a single component or a single test:

.. code-block:: console

    poetry run pytest AIAgent/tests/test_fixtures.py
    poetry run pytest AIAgent/tests/test_onnx.py::TestONNXConversion -sv

Migration note
--------------

The pre-existing component tests are tagged coarsely: ``integration`` for the
in-process suites and ``e2e`` for the full ``runstrat`` pipeline. Their resource
paths are resolved relative to the test file (never the current working
directory), so the whole repository can be collected from the root. CI selects
the markers explicitly per component (``-m "not e2e"`` for the AIAgent and
compstrat jobs, ``-m e2e`` for the runstrat job), so the global default filter
never hides them. Phase 5 (#563) refines this by splitting each component suite
into ``unit/`` and ``integration/`` directories, splitting large resources, and
adding the coverage ratchet.

Fixtures
--------

Shared fixtures live in a ``conftest.py`` next to the component they serve: the
``AIAgent/conftest.py`` fixtures are visible to every test under ``AIAgent/``,
and similarly for ``tools/compstrat/`` and ``tools/runstrat/``.

``AIAgent/conftest.py``
    ``cpu_device`` (autouse) forces ``GeneralConfig.DEVICE`` to CPU,
    ``seeded_rng`` (autouse) seeds ``random``/``numpy``/``torch``,
    ``gamestate_factory`` builds a synthetic ``GameState``,
    ``hetero_factory`` converts it through the production
    ``convert_input_to_tensor`` (with feature-width overrides for legacy
    models), and ``tmp_dataset`` yields an empty ``TrainingDataset`` in a
    temporary directory.

``tools/compstrat/conftest.py``
    Resource-directory and mock-run DataFrame fixtures.

``tools/runstrat/conftest.py``
    Temporary artifacts directory and the built ``ForTests`` map paths.

Adding a fixture
    Put it in the ``conftest.py`` of the component it serves, resolve resources
    relative to ``__file__`` (never the current working directory), and prefer
    building real project objects through production code over hand-rolled
    copies.

Writing a test
--------------

- Prefer the cheapest tier that can catch the regression: cover pure logic with
  plain unit tests (no I/O beyond ``tmp_path``) before reaching for a fixture or
  an integration test.
- Put the test under the component it exercises (``AIAgent/tests``,
  ``tools/*/tests``).
- Mark it with exactly one of ``unit``/``integration``/``e2e`` (plus any
  orthogonal ``gpu``/``network``/``slow``/``serial`` marker). An unmarked test
  still runs in the fast tier, so an omitted marker silently changes the tier it
  appears in.
- Depend on the fixtures from your component's ``conftest.py`` and build project
  objects through the production code paths so they cannot drift from the real
  schema.
- Never open sockets in a unit test: drive the pure parsing helpers directly
  and patch process-global switches such as ``FeatureConfig.DISABLE_MESSAGE_CHECKS``
  with ``monkeypatch``.
- Write anything that touches disk under the ``tmp_path`` fixture.
- Keep it deterministic: the autouse ``seeded_rng`` and ``cpu_device`` fixtures
  make the ``AIAgent`` suite reproducible, but a test must still not rely on
  wall-clock time or on iteration order.

Testability rules
-----------------

- **No filesystem side effects at import.** Importing a module must never
  create, move or delete files. Logging configuration, directory creation and
  similar work belongs in ``main()`` (or the ``if __name__ == "__main__"``
  guard). This keeps imports safe for tests, tooling and documentation builds;
  ``AIAgent/tests/unit/test_import_side_effects.py`` enforces it for the
  runnable AIAgent scripts.
- **Device selection is lazy.** The torch device comes from
  ``config.get_device()`` at call time, never from a module-level constant, so
  importing AIAgent code never probes CUDA and tests can force CPU by patching
  ``torch.cuda.is_available``.
- **Never depend on the current working directory.** Resolve resource paths
  from ``__file__`` (or ``AIAgent/paths.py``), and make filesystem seams
  injectable so tests can redirect them.

Coverage
--------

``pytest-cov`` is installed and ``make test-cov`` produces a terminal and XML
report. There is no threshold yet: Phase 5 (#563) measures the baseline and
turns it into a ratchet (``--cov-fail-under`` raised per pull request) so
coverage never regresses without a deliberate decision.
