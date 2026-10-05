"""Runtime unit tests for ``Connector`` driven by the fake websocket.

These exercise the send/receive loop without a real server: the fake ws records
outgoing frames and replays queued incoming frames. Message parsing itself is
covered by ``test_messages.py``; here we assert start-on-init, GameOver
propagation, connection-loss translation and the reward step counter.
"""

import json

import pytest
from common.game import GameMap
from config import FeatureConfig
from connection.errors_connection import ConnectionLostError
from connection.game_server_conn.connector import Connector

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


def _connector(ws) -> Connector:
    return Connector(ws, _game_map())


def _queued(ws, message_type: str, body: dict | None = None) -> None:
    payload: dict[str, object] = {"MessageType": message_type}
    if body is not None:
        payload["MessageBody"] = body
    ws.incoming.append(json.dumps(payload))


@pytest.fixture(autouse=True)
def _disable_message_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FeatureConfig, "DISABLE_MESSAGE_CHECKS", True)


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
