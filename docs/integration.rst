:description: Integrate a new symbolic engine and convert a trained model to ONNX.
:group: Guides

Integration
===========

.. _integrate-a-new-symbolic-machine:

Integrate a new symbolic machine
--------------------------------

The agent talks to a symbolic virtual machine (SVM) over a websocket protocol
and exchanges JSON messages. To integrate a new engine:

- Implement the server side of the websocket protocol. The agent-side
  reference is ``AIAgent/connection/game_server_conn/`` (``connector.py`` and
  ``messages.py``); the game loop that drives the connection lives under
  ``AIAgent/ml/validation/coverage/game_managers/``.

- Provide serialization and deserialization of the data according to that
  protocol (see ``AIAgent/connection/game_server_conn/messages.py``).

- Implement two execution methods on the engine side:

  - Symbolic execution in training mode (example:
    ``GameServers/VSharp/VSharp.ML.GameServer.Runner/Main.fs``).

  - Execution guided by a trained model (example:
    ``GameServers/VSharp/VSharp.Explorer/AISearcher.fs``).

Integration examples live under ``GameServers/``. Two machines are integrated:

- `V# <https://github.com/PySymGym/VSharp>`__ (a .NET SVM) and
  `its maps <https://github.com/PySymGym/PySymGym/tree/main/maps/DotNet>`__.

- `usvm <https://github.com/PySymGym/usvm>`__ (a JVM SVM) and
  `its maps <https://github.com/PySymGym/PySymGym/tree/main/maps/Java>`__.

V# is the primary game server. A typical end-to-end workflow is automated in
``.github/workflows/build_and_run.yaml``, whose pipeline steps live in the
shared reusable workflow ``.github/workflows/e2e_build_and_run.yml``.

.. _onnx-conversion:

ONNX conversion
---------------

To convert a PyTorch model to ONNX, run ``onyx.py``:

.. code-block:: console

    cd AIAgent
    python3 onyx.py --sample-gamestate <game_state0.json> \
        --pytorch-model <path_to_model>.pth \
        --savepath <converted_model_save_path>.onnx \
        --import-model-fqn <model.module.fqn.Model> \
        --model-kwargs <yaml_with_model_args>.yml \
        [optional] --verify-on <game_state1.json> <game_state2.json> <game_state3.json> ...

- ``pytorch-model`` is the path to the PyTorch model weights.
- Examples of the ``model-kwargs`` YAML file, verification game states, and a
  ``sample-gamestate`` (any can be used) are under ``resources/onnx/``.
- ``import-model-fqn`` is the dotted path to the model class to convert, for
  example ``ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2Parametrized.model.StateModelEncoder``.

See :ref:`guide-symbolic-execution` for how to run execution with the
converted model.