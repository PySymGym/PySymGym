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

The configuration lives in ``[tool.pytest.ini_options]`` in the root
``pyproject.toml``.

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

