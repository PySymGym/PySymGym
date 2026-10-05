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
    models), ``tmp_dataset`` yields an empty ``TrainingDataset`` in a
    temporary directory, ``fake_websocket`` returns a recording, no-socket
    websocket for the connection tests, and ``fake_namespace``/``fake_proc``
    provide the game managers' multiprocessing namespace and a
    ``subprocess.Popen`` stand-in.

``tools/compstrat/conftest.py``
    Resource-directory and mock-run DataFrame fixtures.

``tools/runstrat/conftest.py``
    Temporary artifacts directory and the built ``ForTests`` map paths.

``tools/dataset_tools`` and ``tools/util`` now have ``tests/unit/`` roots as
well. They need no conftest of their own: the root ``conftest.py`` puts every
tool directory on ``sys.path``, so ``clean``, ``generate_episodes`` and
``parse_pretty`` import as top-level modules.

``AIAgent/tests/unit/ml/`` is the unit root for the ML layer (dataset
transforms, training and experiment helpers, model-forward smoke tests) and
``AIAgent/tests/unit/ml/validation/`` its orchestration subtree (game managers
and the coverage-validation flow). They reuse the ``AIAgent/conftest.py``
fixtures and build every graph input through
``ml.dataset.convert_input_to_tensor`` so the tests cannot drift from the
production tensor schema.

Model smoke tests forward a synthetic ``hetero_factory`` graph through every
model that still matches the production tensor schema (currently
``NorthernPenguin`` and ``InvisibleCow``: game width 7 and six state features),
asserting output dtype/shape and the ``log_softmax`` normalization;
``modelop.filemanager.save_model`` is exercised with ``tmp_path`` and a patched
``datetime``. Legacy architectures that cannot consume the current tensors are
tracked in issues #571/#572 instead of being tested.

Adding a fixture
    Put it in the ``conftest.py`` of the component it serves, resolve resources
    relative to ``__file__`` (never the current working directory), and prefer
    building real project objects through production code over hand-rolled
    copies.

Fakes and sockets
-----------------

Orchestration seams are tested with fakes, never real I/O:

- **Sockets.** Replace ``socket.socket``, ``httplib2.Http``, ``psutil`` and
  ``websocket.WebSocket`` with recording fakes and drive the real client code.
  The ``fake_websocket`` fixture records outgoing frames in ``sent`` and
  replays queued ``incoming`` frames on ``recv`` (a queued exception is raised
  instead of returned), so ``Connector``'s start/step/reward loop can be
  asserted without a server. ``game_server_socket_manager`` must call
  ``return_instance`` on **both** the success and failure paths — assert it
  explicitly rather than trusting the ``finally``.
- **Subprocesses and files.** For the model game manager, patch
  ``subprocess.Popen`` (use the ``fake_proc`` fixture), the port helper
  (``common.network_utils.look_for_free_port_locked``), ``delete_dir`` and the
  module path constants; point ``svms_output_path`` at ``tmp_path`` and write
  the ``{MapName}result`` file directly. The manager is built with
  ``fake_namespace`` (a real lock, no ``multiprocessing.Manager``).
- **Multiprocessing.** The each-step manager and ``ValidationCoverage`` are
  tested by patching ``Connector`` (the fake must expose ``GameOver``) and
  ``game_server_socket_manager``; for the public ``validate_coverage`` loop,
  patch ``multiprocessing.Manager``/``Pool`` and ``tqdm`` and mark the test
  ``serial``. A fake ``BaseGameManager`` subclass plus a fake dataset cover the
  manager-unset, GameFailed, missing-steps and exception branches without
  spawning a worker.
- **Cleanup invariants.** When a function acquires a resource (a socket, an
  instance, a process), test the success path and an exception in the body, and
  assert the release call in each.

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
- For graph/dataset transforms, build the input through
  ``ml.dataset.convert_input_to_tensor`` (via ``gamestate_factory`` /
  ``hetero_factory``) and assert exact tensor contents; the ``tmp_dataset``
  fixture yields an empty ``TrainingDataset`` for the filtering helpers.
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
