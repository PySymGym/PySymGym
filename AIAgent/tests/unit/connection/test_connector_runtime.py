"""Runtime unit tests for ``Connector`` driven by the fake websocket.

These exercise the send/receive loop without a real server: the fake ws records
outgoing frames and replays queued incoming frames. Message parsing itself is
covered by ``test_messages.py``; here we assert start-on-init, GameOver
propagation, connection-loss translation and the reward step counter.
"""

import json

import pytest
import websocket
from common.config.feature_config import FeatureConfig
from common.game import GameMap
from connection.errors_connection import ConnectionLostError
from connection.game_server_conn.connector import Connector
from connection.game_server_conn.messages import PROTOCOL_VERSION

pytestmark = [pytest.mark.unit, pytest.mark.serial]


def _game_map() -> GameMap:
    return GameMap(
        StepsToPlay=10,
        StepsToStart=0,
        AssemblyFullName="assembly",
        NameOfObjectToCover="Method",
        DefaultSearcher="BFS",
        MapName="Method_0",
    )


def _connector(ws, feature_config: FeatureConfig | None = None) -> Connector:
    # The unchecked (BareObject) decode is the default here: the runtime
    # loop tests are about the protocol flow, not strict validation (the
    # strict path is pinned by the two tests that pass an explicit config).
    return Connector(
        ws, _game_map(), feature_config or FeatureConfig(disable_message_checks=True)
    )


def _queued(
    ws,
    message_type: str,
    body: dict | None = None,
    version: int | None = PROTOCOL_VERSION,
) -> None:
    # The default simulates a v2 server (the envelope carries the
    # Version field); pass version=None for a legacy v1 payload.
    payload: dict[str, object] = {"MessageType": message_type}
    if version is not None:
        payload["Version"] = version
    if body is not None:
        payload["MessageBody"] = body
    ws.incoming.append(json.dumps(payload))


def test_connector_sends_start_message_on_init(fake_websocket) -> None:
    ws = fake_websocket

    connector = _connector(ws)

    assert connector._current_step == 0
    assert connector.game_is_over is False
    assert connector.map.MapName == "Method_0"
    assert len(ws.sent) == 1
    start = json.loads(ws.sent[0])
    assert start["MessageType"] == "start"
    assert json.loads(start["MessageBody"])["MapName"] == "Method_0"


def test_recv_state_returns_the_message_body(fake_websocket) -> None:
    ws = fake_websocket
    connector = _connector(ws)
    _queued(
        ws,
        "ReadyForNextStep",
        {
            "GraphVertices": [],
            "States": [],
            "PathConditionVertices": [],
            "Map": [],
        },
    )

    body = connector.recv_state_or_throw_gameover()

    assert body.GraphVertices == []


def test_gameover_message_raises_with_counts(fake_websocket) -> None:
    ws = fake_websocket
    connector = _connector(ws)
    _queued(
        ws,
        "GameOver",
        {"ActualCoverage": 55, "TestsCount": 6, "StepsCount": 7, "ErrorsCount": 2},
    )

    with pytest.raises(Connector.GameOver) as exc_info:
        connector.recv_state_or_throw_gameover()

    assert connector.game_is_over is True
    assert exc_info.value.actual_coverage == 55
    assert exc_info.value.tests_count == 6
    assert exc_info.value.steps_count == 7
    assert exc_info.value.errors_count == 2


def test_connection_reset_becomes_connection_lost(fake_websocket) -> None:
    ws = fake_websocket
    connector = _connector(ws)
    ws.incoming.append(ConnectionResetError())

    with pytest.raises(ConnectionLostError):
        connector.recv_state_or_throw_gameover()


@pytest.mark.parametrize(
    "exc_type",
    [
        websocket.WebSocketConnectionClosedException,
        websocket.WebSocketTimeoutException,
    ],
)
def test_real_websocket_exceptions_become_connection_lost(
    fake_websocket, exc_type: type[websocket.WebSocketException]
) -> None:
    ws = fake_websocket
    connector = _connector(ws)
    ws.incoming.append(exc_type())

    with pytest.raises(ConnectionLostError) as exc_info:
        connector.recv_state_or_throw_gameover()

    # The translation chains the real websocket exception as the cause.
    assert isinstance(exc_info.value.__cause__, exc_type)


def test_send_step_serializes_and_records_state(fake_websocket) -> None:
    ws = fake_websocket
    connector = _connector(ws)

    connector.send_step(next_state_id=3, predicted_usefullness=42)

    assert connector._sent_state_id == 3
    step = json.loads(ws.sent[-1])
    assert step["MessageType"] == "step"
    assert json.loads(step["MessageBody"])["StateId"] == 3


def test_move_reward_increments_step(fake_websocket) -> None:
    ws = fake_websocket
    connector = _connector(ws)
    _queued(ws, "MoveReward", {})

    connector.recv_reward_or_throw_gameover()

    assert connector._current_step == 1


def test_incorrect_predicted_state_does_not_increment_step(fake_websocket) -> None:
    ws = fake_websocket
    connector = _connector(ws)
    connector.send_step(next_state_id=3, predicted_usefullness=42)
    _queued(ws, "IncorrectPredictedStateId", {})

    connector.recv_reward_or_throw_gameover()

    assert connector._current_step == 0


def test_strict_validation_decodes_incorrect_predicted_state(fake_websocket) -> None:
    # Pre-#582, the strict path decoded the body as Reward and raised
    # DeserializationException on this normal protocol event.
    ws = fake_websocket
    connector = _connector(ws, FeatureConfig(disable_message_checks=False))
    connector.send_step(next_state_id=3, predicted_usefullness=42)
    _queued(ws, "IncorrectPredictedStateId", {"StateId": 9})

    body = connector.recv_reward_or_throw_gameover()

    assert connector._current_step == 0
    assert body.StateId == 9


def test_strict_validation_decodes_move_reward(fake_websocket) -> None:
    ws = fake_websocket
    connector = _connector(ws, FeatureConfig(disable_message_checks=False))
    _queued(
        ws,
        "MoveReward",
        {
            "ForMove": {"ForCoverage": 1, "ForVisitedInstructions": 2},
            "MaxPossibleReward": 3,
        },
    )

    body = connector.recv_reward_or_throw_gameover()

    assert connector._current_step == 1
    assert body.ForMove.ForCoverage == 1
    assert body.MaxPossibleReward == 3


def test_legacy_versionless_envelope_is_still_accepted(fake_websocket) -> None:
    # A v1 server (no Version field) is tolerated: the message decodes
    # and the once-per-process legacy warning is logged, not an error.
    ws = fake_websocket
    connector = _connector(ws)
    _queued(
        ws,
        "ReadyForNextStep",
        {
            "GraphVertices": [],
            "States": [],
            "PathConditionVertices": [],
            "Map": [],
        },
        version=None,
    )

    body = connector.recv_state_or_throw_gameover()

    assert body.GraphVertices == []
