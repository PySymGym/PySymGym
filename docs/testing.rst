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

    poetry run pytest AIAgent/tests/unit/test_fixtures.py
    poetry run pytest AIAgent/tests/integration/test_onnx.py::TestONNXConversion -sv

Layout
------

Each component keeps its own test tree, split by tier into a directory named
after the marker its tests carry:

- ``unit/`` — the fast, isolated tests (``AIAgent/tests/unit/``,
  ``tools/*/tests/unit/``).
- ``integration/`` — the in-process tests that need golden fixtures
  (``AIAgent/tests/integration/``, ``tools/compstrat/tests/integration/``).
- ``e2e/`` — the full pipelines (``tools/runstrat/tests/e2e/``).

A component only creates the tier directories it needs. The ``tests/`` tree at
the repository root holds the cross-component infrastructure tests.

The marker is the source of truth, not the directory: CI selects markers
explicitly per component (``-m "not e2e"`` for the AIAgent and compstrat jobs,
``-m e2e`` for the runstrat job) and the root default filter deselects
``integration`` and ``e2e``. A module may declare its tier once at module level
(``pytestmark = pytest.mark.unit``) or per function; every ``test_*`` function
must resolve to exactly one tier, which ``tests/test_pytest_infra.py`` enforces
statically together with the tier directory (``unit/``, ``integration/`` or
``e2e/``) it lives in.

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
    models), ``step_factory`` builds a processed step (the
    ``hetero_factory`` graph with a given ``y_true``), ``serial_pool``
    replaces ``multiprocessing.Pool`` with an in-process serial stand-in
    (forked workers are invisible to pytest-cov), ``tmp_dataset`` yields an
    empty ``TrainingDataset`` in a temporary directory, ``fake_websocket``
    returns a recording, no-socket websocket for the connection tests, and
    ``fake_namespace``/``fake_proc`` provide the game managers'
    multiprocessing namespace and a ``subprocess.Popen`` stand-in.

``tools/compstrat/conftest.py``
    Resource-directory and mock-run DataFrame fixtures.

``tools/runstrat/conftest.py``
    Temporary artifacts directory and the built ``ForTests`` map paths.

``AIAgent/tests/integration/conftest.py``
    ``resources_dir`` exposes the absolute ``integration/resources/`` directory,
    and its ``pytest_generate_tests`` hook parametrizes the config-driven tests
    (the ONNX model configs and the SVMs validation configs) from that
    directory, so no test computes a resource path itself.

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
production tensor schema. The malformed-input branches of
``convert_input_to_tensor`` are pinned by mutating a ``gamestate_factory``
copy (exact error type and message, the ``(None, None)`` return) rather than
hand-rolled gamestates. The ``TrainingDataset`` lifecycle tests
(``test_dataset_lifecycle.py``) cover the in-memory (``load_to_cpu``) and
on-disk loading paths, the train/validation index switching, and the
metadata refresh; they pin the stale ``len()`` before the first
``switch_to`` and drive the auto-processing run at the end of
``__init__`` through the ``serial_pool`` fixture. The processing and
maintenance tests (``test_dataset_maintenance.py``) build the raw dataset
layout (per map a ``result`` file plus ``<id>_gameState`` /
``<id>_movedState`` step files) and drive ``process()`` through the
``serial_pool`` fixture as well. They also walk the ``update_map`` branch
matrix (new map, below-threshold result-only save, merge, reproduced,
replace, and the two no-op cases) with ``step_factory`` steps, pinning the
exact ``logging.info`` messages via caplog (including the trailing space of
the merge/reproduce messages) and the ``load_to_cpu`` cache bookkeeping.
They also exercise ``_get_map_steps`` (sorted ``.pt`` load, ``result``
excluded, default device) and ``_merge_steps`` directly (graph-match merge
math, unique-step append order, new vertex-count groups). The
``merge_distributions`` device-mismatch branch (3 statements) is not
coverable on a CPU-only machine — meta tensors cannot be copied out — so
``ml/dataset.py`` tops out at 99.0%. The protocol tests
(``test_protocols.py``) pin the ``ml.protocols`` abstract interfaces:
abstractness, the abstract-method set, a concrete subclass satisfying the
interface, and the abstract bodies raising ``NotImplementedError`` when
invoked unbound. The training-loop tests (``test_train.py``) pin
``ml.training.train`` with a real small model, optimizer and criterion: the
``PYTORCH_CUDA_ALLOC_CONF`` expandable-segments switch, the ``y_true``
clamping to ``{0, 1}`` before the criterion sees it, the per-batch optimizer
step, and the skip of backward/step when the criterion returns zero. The
experiment-helper tests (``test_experiments_utils.py``) pin the remaining
``ml.training.experiments_utils`` helpers: ``get_model`` reloading a saved
state dict with the ``lin_last`` weights overwritten by the exact seeded
random values, ``find_unfinished_maps`` printing the started-but-not-finished
maps parsed from a server log (per port, a ``Finish`` without a ``Start``
prints nothing), ``sync_dataset_with_description`` pruning dataset dirs
missing from the map description, ``find_dominators_in_cfg`` appending the
dominator edges (type 3, dominator to dominated, self-loops skipped) per
entry point and raising ``ValueError`` without one, and ``sample_dataset``
copying the rounded percentage of ``.pt`` files (with the map's ``result``
file on first copy) from a two-levels-deep relative dataset. The inference
tests (``test_predict_wrapper.py``) pin ``ml.predict.predict_state_with_dict``
with a fixed-output fake model (the exact max-row-sum selection, the
state-map remapping, the ``no_grad`` output) plus one real-model run, and
``ml.training.wrapper.TrainingModelWrapper`` (the name/model accessors, the
initial ``model_copy`` alias, the ``make_copy`` deep copy and rename, the
``predict`` delegation to ``predict_state_with_dict`` with the production
converted graph and state map, and a real-model ``predict`` run). The
pretraining tests (``test_het_gnn_test_train.py``) pin
``ml.het_gnn_test_train.HetGNNTestTrain`` as a seam: ``train_and_save`` runs
against a fake model class (one parameter, output ``weight + 0.5``) and
pickled synthetic graphs, pinning the 85/15 split counts, the exact per-epoch
loss line (a reference ``Adam`` step computed in the test), the forward-call
count, and the ``<module>/<hidden>ch/<epochs>e`` checkpoint layout with both
the state-dict and full-model files; the ``BALANCE_DATASET`` truncation is
pinned by monkeypatching the module constant with one 5001-element file
(marked ``slow``). ``train`` / ``tst`` / ``save_simple`` are driven directly
against the reference math. The loss-validation tests
(``AIAgent/tests/unit/ml/validation/test_validate_loss.py``) pin
``ml.validation.loss.validate_loss`` with a real ``TrainingDataset`` (steps
written through ``step_factory``) and a fixed-output fake model: the exact
averaged ``MSELoss`` over the ``{0, 1}``-thresholded ``y_true`` (the same run
would average 0.135 without the thresholding), the per-batch loss sequence,
and the ``DataLoader`` (``batch_size``, ``shuffle=False``) and ``tqdm``
(``desc``/``ncols``/``colour``) kwargs. The ``ModelGameManager`` play-flow
tests (``test_process_game_manager.py``) pin ``_play_game_map`` with
instance-level fakes for ``_run_game_process`` / ``_get_result`` and the
``fake_proc`` fixture: the ``GameResult`` / ``GameFailed`` wrapping into
``GameResultDetails`` / ``GameFailedDetails`` (with the
``SVMConnectionInfo``), the exact
``"Failed to instantiate process for <map>"`` message, the timeout and
exception branches killing the process and storing the exception CLASS as
the ``GameFailed`` reason, and the no-process warning; the preparator is
pinned (clean-then-export order, the existing-dir delete, the
no-op-when-missing) and the full-coverage-without-oracle warning of
``_get_result`` is pinned via caplog. The ``EachStepGameManager`` tests
(``test_each_step_game_manager.py``) pin the remaining branches: the
``FunctionTimedOut`` branch (the model is saved, the reason is the
exception class, the exact warning) driven by shadowing the
``func_set_timeout``-decorated ``_play_game_map_with_svm`` with an
instance attribute, the ``GameInterruptedError`` branch (no save, the exact
``"<name> failed on map <map> with <class>: <desc>"`` message), the
``func_timeout`` swallow quirk (a ``FunctionTimedOut`` raised from inside
the wrapped function is discarded by its worker thread, the method returns
``None`` and the unpacking ``TypeError`` is handled by the generic branch),
the hardcoded ``42.0`` predicted usefulness, ``delete_game_artifacts``
(drop and no-op), the ``notify_steps_requirement`` no-op, and the
preparator's ``prepare``.

The entrypoint seam tests (``AIAgent/tests/unit/test_run_training.py``) pin
``run_training.py`` without end-to-end runs: ``run_training()`` is driven once
with recording fakes for ``TrainingDataset``, ``optuna``, ``mlflow``,
``joblib`` and ``objective``, and the ``objective`` partial it wires up is
inspected to extract the ``model_init`` / ``validate`` / ``criterion_init``
closures, which are then driven directly. The study branch pins the
study-level seed, the dataset construction from the module path constants,
the TPE sampler and study creation, and the per-trial callback (joblib dumps
of the study and trial, re-entering the active run, logging both artifacts,
tagging the best trial number); the trial-replay branch (``trial_uri`` set)
pins the artifact download to ``REPORT_PATH``, the joblib load, and the
replay of the loaded trial ``n_trials`` times without creating a fresh
study. The ``validate`` closures are pinned per mode: the criterion mode
switches the dataset to the configured split, drives ``validate_loss`` with
a fresh ``BCEWithLogitsLoss(reduction="sum")``, and names the metric
``BCEWithLogitsLoss__<DATASET MODE>``; the SVM mode writes the per-trial
``svms_result_table.csv`` (sorted map-name header, written once), drives
``ValidationCoverage`` and ``get_svms_statistics``, logs the table, and
raises ``RuntimeError("Validation failed")`` on a failed map when
``fail_immediately`` is set; the custom mode runs the sequence in order,
merges the metrics, and returns the first stage's result. Each mode also
pins its ``PYTORCH_CUDA_ALLOC_CONF`` expandable-segments switch. The
per-trial ``objective`` loop is pinned with a scripted trial and a recording
model: the per-trial seed derivation, the exact optuna search space, the
``EarlyStopping`` construction, the ``model_init`` kwargs drawn from the
trial settings, the trial artifact layout (``model_kwargs.yaml``, per-epoch
checkpoints, mlflow params/metrics), the per-epoch train/validate sequence,
the early-stop break (with its print), and the ``dynamic_dataset`` metadata
refresh. ``main()`` is pinned with a production-shaped config YAML (alias
keys): the parsed config forwarded to ``run_training``, the spawn start
method, the MLflow setup (tracking uri only when set, experiment, tags,
system metrics), the gpu-monitor logger silenced to ``ERROR``, the printed
device, and the ``Config`` validator rejecting ``weights_uri`` without
``trial_uri``. The ``model_init`` checkpoint loading is pinned through that
seam, including the ``map_location`` fix for CUDA-saved checkpoints on
CPU-only machines (#667, regression test requires CUDA and skips otherwise).
The companion entrypoint seam tests
(``AIAgent/tests/unit/test_launch_servers.py``) pin ``launch_servers.py``
the same way: ``wait_for_port`` with a scripted socket, the four broker
handlers with fake requests and queues (the dequeue success / empty /
failure-requeue paths, the enqueue kill-and-rewrap versus keep-pid paths,
the results append / recv), ``run_server_instance`` with a fake ``Popen``,
a pinned port and a fresh port lock, the ``run_servers`` placeholders,
``kill_server`` / ``kill_process`` with fake ``os.kill`` / ``psutil`` (the
missing-process warnings, the zombie shortcut, the never-dies
``RuntimeError``), ``server_manager`` (queue on enter, kill-all on exit),
and ``main()`` (the parsed ``process_count`` handed to ``server_manager``
with a fresh queue, the four declared broker routes on the application,
the broker port, and the module-global reset).

Model smoke tests forward a synthetic ``hetero_factory`` graph through every
model that still matches the production tensor schema (currently
``NorthernPenguin`` and ``InvisibleCow``: game width 7 and six state features),
asserting output dtype/shape and the ``log_softmax`` normalization;
``modelop.filemanager.save_model`` is exercised with ``tmp_path`` and a patched
``datetime``. Legacy architectures that cannot consume the current tensors are
tracked in issues #571/#572 instead of being tested.

The runstrat CLI tests (``tools/runstrat/tests/unit/``) pin the benchmarking
tool without end-to-end strategy runs. The parsing tests (``test_parsing.py``)
pin ``parse_runner_output`` (the exact ``(total_time_sec, tests, errors,
coverage)`` tuple from a runner transcript, and the ``AttributeError`` with its
``__notes__`` on a malformed transcript), ``parse_prebuilt`` (class-qualified
versus top-level method expansion), and the file-reading seams ``load_config``
(a ``dll,method`` CSV into ``LaunchInfo`` rows, the ``Path``-typed ``dll``
field) and ``load_prebuilt_config`` (a JSON config structured into
``PrebuiltConfig`` and expanded). The ``subprocess_calls`` tests
(``test_subprocess_calls.py``) pin the exact dotnet argv (the base flags, the
``--model`` suffix for the AI strategy, the ``cwd``) and the
exception-propagation branches (``CalledProcessError`` and
``FunctionTimedOut`` re-raised unchanged); the timeout child-cleanup defect is
#590-tracked and not asserted.

Adding a fixture
    Put it in the ``conftest.py`` of the component it serves, resolve resources
    relative to ``__file__`` (never the current working directory), and prefer
    building real project objects through production code over hand-rolled
    copies.

Integration fixtures live under ``<component>/tests/integration/resources/`` and
reach the tests only through the component conftest (an absolute path resolved
from ``conftest.py``), never through a path relative to the current working
directory. Production inputs a test also consumes — such as the ONNX reference
gamestates under ``resources/onnx/`` — are read through ``AIAgent/paths.py``
instead of being copied into the test tree.

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
  ``tools/*/tests``) and in the directory named after its tier (``unit/``,
  ``integration/`` or ``e2e/``).
- Mark it with exactly one of ``unit``/``integration``/``e2e`` (plus any
  orthogonal ``gpu``/``network``/``slow``/``serial`` marker). The directory must
  match the tier marker; an unmarked test still runs in the fast tier, so an
  omitted marker silently changes the tier it appears in.
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

Coverage is measured over the production packages (``--cov=AIAgent
--cov=tools``) with ``pytest-cov``. The ``tests/`` trees, ``conftest.py`` files
and the test helpers are excluded from the measurement, as are ``__main__``
guards, so the number reflects the code under test rather than the tests
themselves.

The gated tier is unit + integration:

.. code-block:: console

    poetry run pytest -m "not e2e" --cov=AIAgent --cov=tools --cov-report=term-missing
    make test-cov     # same tier, plus XML and JSON reports

The threshold is a ratchet: ``[tool.coverage.report] fail_under`` in
``pyproject.toml`` is set to the measured baseline, floored to a whole percent.
A pull request that drops coverage below it fails CI
(``.github/workflows/python_tests.yaml``). Raise the value in the same pull
request that improves coverage; never lower it without a deliberate decision
recorded in the pull request.

Exclusions
~~~~~~~~~~

``AIAgent/ml/models/RGCNEdgeTypeTAG3VerticesDoubleHistory2/model.py`` is
omitted from the measurement (``[tool.coverage.run] omit``): the legacy
architecture cannot consume the current graph tensors — the 2-column history
``edge_attr`` is passed as a 1-D ``edge_weight`` (#571) and the three legacy
models cannot forward or instantiate with the current tensors (#572). It is
tracked in those issues instead of being tested, and its removal is pending
#581 (model registry / dead-zoo cleanup), which also removes the omit line.
