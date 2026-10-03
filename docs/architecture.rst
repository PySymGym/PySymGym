Architecture
============

*High-level map of the system and where to find things; the starting point for
navigating the code.*

What PySymGym is
----------------

PySymGym couples a Python reinforcement-learning agent with symbolic execution
engines (symbolic virtual machines, SVMs). Path selection is modeled as a game:
the current state of the symbolic execution process, represented as an
interprocedural control flow graph equipped with execution metadata, is the
game map; symbolic states are the pieces the agent moves. For each step the
agent observes the map, selects a state, and sends it to the game server; the
server performs the step and returns the updated map. Depending on the scoring
function, the agent can aim for full coverage in a minimal number of moves, or
with a minimal number of generated tests, or something else. All data,
including the map, crosses the boundary as JSON, so the gym is not tied to any
single symbolic engine.

Components
----------

The repository separates the engine, the agent, the inputs, and the tooling.

``AIAgent/``
    The Python side: the game protocol, machine-learning models, training, and
    inference. Its subpackages are:

    ``common/``
        Shared domain model (``GameMap``/``GameState``), typed config, and
        general utilities.
    ``connection/``
        Transport to game servers (a websocket broker) and the protocol
        message types.
    ``ml/``
        Datasets, graph-neural-network models, training (including Optuna
        hyper-parameter tuning), validation, and inference.

    Main entrypoints: ``run_training.py`` (training and tuning),
    ``launch_servers.py`` (server manager), and ``onyx.py`` (PyTorch-to-ONNX
    conversion and verification).

``GameServers/``
    Symbolic execution engines extended to speak the game protocol, kept as
    git submodules. V# (.NET) is the primary engine; usvm covers the JVM.

``maps/``
    Target programs used as symbolic-execution inputs and as training data,
    mostly git submodules.

``tools/``
    Standalone command-line tools: ``runstrat`` (benchmark a strategy),
    ``compstrat`` (compare two strategies), ``dataset_tools`` (expand and
    clean datasets), and ``util``.

``configs/``
    Ready-to-run training and validation YAML configurations.

``workflow/``
    Test-workflow configurations and dataset templates referenced by the README
    and CI.

Data and control flow
---------------------

A training run starts from an initial dataset produced by a baseline path
strategy, then alternates two stages: hyper-parameter tuning to get the best
model for the current dataset, and dataset improvement in which that model
guides SVM runs and contributes better episodes. Tuning and dataset
improvement are repeated until the desired result is reached, after which the
trained model can guide symbolic execution directly. The sequence is
illustrated by ``resources/game_process_illustration.png`` in the repository
root.

Where to look for X
-------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Need
     - Look at
   * - Add or change an ML model
     - ``AIAgent/ml/models/``
   * - Change training or tuning
     - ``AIAgent/run_training.py``, ``AIAgent/ml/training/``
   * - Change validation or statistics
     - ``AIAgent/ml/validation/``
   * - Change the game protocol
     - ``AIAgent/connection/``, ``AIAgent/common/game.py``
   * - Integrate a new symbolic engine
     - ``GameServers/``, ``AIAgent/connection/``
   * - Add a target program
     - ``maps/``
   * - Benchmark or compare strategies
     - ``tools/runstrat/``, ``tools/compstrat/``
   * - Maintain the dataset
     - ``tools/dataset_tools/``
   * - Run or tune a configuration
     - ``configs/``, ``workflow/``
   * - Understand or change CI
     - ``.github/workflows/``
