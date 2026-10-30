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

.. _wire-protocol:

Wire protocol
-------------

This section is the wire contract for engine integrators: the message
types, the field lists, the encoding rules, and the versioning. The
executable schema is the message dataclasses in
``AIAgent/connection/game_server_conn/messages.py`` (with the domain types
in ``AIAgent/common/game.py``); the two must stay in agreement — the
wire-format pins in ``AIAgent/tests/unit/connection/`` are the drift net.

Transport and framing
~~~~~~~~~~~~~~~~~~~~~

The agent and the game server talk over a websocket and exchange one JSON
object per text frame. Every message is an envelope with a ``MessageType``
(string) and a ``MessageBody``:

.. code-block:: json

    {"MessageType": "step", "MessageBody": "{\"StateId\": 3, \"PredictedStateUsefulness\": 1.5}"}

Protocol versions
~~~~~~~~~~~~~~~~~

- **v1** (legacy): the envelope has no ``Version`` field. This is the
  format the V# and usvm servers speak today.
- **v2**: the envelope carries ``"Version": 2``.

The version the agent speaks is ``PROTOCOL_VERSION`` in
``messages.py`` (currently 2). On receive, the agent:

- accepts a ``Version`` equal to ``PROTOCOL_VERSION``;
- rejects any other present ``Version`` with
  ``ProtocolVersionMismatchError`` (the game fails with that reason);
- accepts a missing ``Version`` as a legacy v1 server and logs a loud
  warning (once per process).

The agent does not emit ``Version`` yet: the current usvm server rejects
unknown envelope keys, so client-side emission ships together with the
server-side rollout (the V# and usvm follow-ups that add the field to
their outgoing messages). Any change to the message types, fields, or
encoding below is a new version: bump ``PROTOCOL_VERSION`` and update this
page in the same change.

Client to server
~~~~~~~~~~~~~~~~

The client sends the ``MessageBody`` as a **JSON string inside the JSON
envelope** (JSON-in-JSON) — see the encoding note below.

- ``start`` — body: the ``GameMap`` fields (``StartMessageBody`` mirrors
  ``common.game.GameMap`` field-for-field): ``StepsToPlay`` (int),
  ``StepsToStart`` (int), ``AssemblyFullName`` (str),
  ``NameOfObjectToCover`` (str), ``DefaultSearcher`` (str, ``"BFS"`` or
  ``"DFS"``), ``MapName`` (str).
- ``step`` — body: ``StateId`` (int), ``PredictedStateUsefulness`` (float).
  In ``svms_each_step`` mode the agent sends the model's predicted value for
  the chosen (argmax) state.
- ``stop`` — body: an empty object. The servers handle it; the client does
  not send it today.

Server to client
~~~~~~~~~~~~~~~~

The server sends the ``MessageBody`` as a **nested JSON object**.

- ``ReadyForNextStep`` — body: ``GameState`` (``GraphVertices``,
  ``States``, ``PathConditionVertices``, ``Map`` — see
  ``common.game.GameState``).
- ``MoveReward`` — body: ``Reward`` (``ForMove`` with ``ForCoverage`` and
  ``ForVisitedInstructions`` (ints), ``MaxPossibleReward`` (int)).
- ``IncorrectPredictedStateId`` — body: ``StateId`` (int). Sent when the
  predicted state does not exist; the agent continues with the server's
  state.
- ``GameOver`` — body: ``ActualCoverage`` (int or null), ``TestsCount``
  (int), ``StepsCount`` (int), ``ErrorsCount`` (int).
- ``ServerError`` — body: ``ErrorMessage`` (str) in V#; usvm sends the key
  ``Message`` instead. The agent does not decode this body today; the
  message surfaces as a decode failure.

Encoding note: the JSON-in-JSON client-body asymmetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The client encodes ``MessageBody`` as a JSON **string** (the server
decodes the string as JSON a second time), while the server encodes it as
a nested JSON **object**. This asymmetry is a known wart of the current
protocol, flagged for a future protocol revision; changing it is a
breaking wire change (a future version).

.. _result-and-steps-files:

Result and steps files
----------------------

In ``svms_model`` (send-model) mode the agent launches the engine as a
subprocess (the platform's ``SVMSInfo.launch_command``, with ``{OutputDir}``
substituted) and collects the outcome from two files the engine writes into
that output directory. The agent polls for the result file until it appears
or the per-map timeout elapses:

- ``<OutputDir>/<MapName>result`` — four space-separated integers::

    actual_coverage_percent tests_count steps_count errors_count

  ``steps_count`` is the **total** trace steps (the startup steps included);
  the agent subtracts ``StepsToStart`` to obtain the model-played count (see
  :ref:`svm-validation-step-metric`).
- ``<OutputDir>/<MapName>_steps`` — a JSON array with one entry per
  model-played step (the game state and the model output). The engine writes
  it only on request: after writing the result file it connects to the
  agent's listening port (the ``{Port}`` substitution) and reads one byte —
  ``1`` = save the steps, ``0`` = skip — and writes the file only for ``1``.

.. _onnx-conversion:

ONNX conversion
---------------

To convert a PyTorch model to ONNX, run ``onyx.py``:

.. code-block:: console

    cd AIAgent
    python3 onyx.py --sample-gamestate <game_state0.json> \
        --pytorch-model <path_to_model>.pth \
        --savepath <converted_model_save_path>.onnx \
        --model-name <model_name> \
        --model-kwargs <yaml_with_model_args>.yml \
        [optional] --verify-on <game_state1.json> <game_state2.json> <game_state3.json> ... \
        [optional] --opset-version <n>

- ``pytorch-model`` is the path to the PyTorch model weights.
- Examples of the ``model-kwargs`` YAML file, verification game states, and a
  ``sample-gamestate`` (any can be used) are under ``resources/onnx/``.
- ``model-name`` is the registry name of the model to convert (see
  ``AIAgent/ml/models/registry.py``), for example ``NorthernPenguin``.
- ``--opset-version`` is the ONNX opset version to export with (default: 17).

See :ref:`guide-symbolic-execution` for how to run execution with the
converted model.