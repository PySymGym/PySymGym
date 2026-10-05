"""Unit tests for the game-server client/server messages."""

import json

import pytest
from config import FeatureConfig

from common.game import GameMap
from connection.game_server_conn.messages import (
    ClientMessage,
    ClientMessageType,
    GameOverServerMessage,
    ServerMessage,
    ServerMessageType,
    StartMessageBody,
    StepMessageBody,
)

pytestmark = pytest.mark.unit


def _game_map() -> GameMap:
    return GameMap(
        StepsToPlay=10,
        StepsToStart=0,
        AssemblyFullName="assembly",
        NameOfObjectToCover="Method",
        DefaultSearcher="BFS",
        MapName="Method_0",
    )


def test_client_message_sets_type_from_start_body() -> None:
    message = ClientMessage(StartMessageBody(**_game_map().to_dict()))
    assert message.MessageType is ClientMessageType.START


def test_client_message_sets_type_from_step_body() -> None:
    message = ClientMessage(StepMessageBody(StateId=3, PredictedStateUsefulness=1.5))
    assert message.MessageType is ClientMessageType.STEP


def test_client_message_json_contains_encoded_body() -> None:
    message = ClientMessage(StepMessageBody(StateId=3, PredictedStateUsefulness=1.5))
    decoded = json.loads(message.to_json())
    assert decoded["MessageType"] == "step"
    assert "StateId" in decoded["MessageBody"]


def test_start_message_body_json_round_trip() -> None:
    body = StartMessageBody(**_game_map().to_dict())
    assert StartMessageBody.from_json(body.to_json()) == body


@pytest.mark.parametrize("disabled", [True, False])
def test_from_json_handle_respects_message_checks(
    disabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FeatureConfig, "DISABLE_MESSAGE_CHECKS", disabled)
    payload = json.dumps({"MessageType": "GameOver"})

    result = ServerMessage.from_json_handle(payload, expected=ServerMessage)

    if disabled:
        assert not isinstance(result, ServerMessage)
        assert result.MessageType == "GameOver"
    else:
        assert isinstance(result, ServerMessage)
        assert result.MessageType is ServerMessageType.GAMEOVER


def test_from_json_handle_raises_on_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FeatureConfig, "DISABLE_MESSAGE_CHECKS", False)
    payload = json.dumps({"MessageType": "GameOver"})

    with pytest.raises(ServerMessage.DeserializationException):
        GameOverServerMessage.from_json_handle(payload, expected=GameOverServerMessage)


def test_game_over_message_exposes_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FeatureConfig, "DISABLE_MESSAGE_CHECKS", False)
    payload = json.dumps(
        {
            "MessageType": "GameOver",
            "MessageBody": {
                "ActualCoverage": 12,
                "TestsCount": 3,
                "StepsCount": 4,
                "ErrorsCount": 1,
            },
        }
    )

    decoded = GameOverServerMessage.from_json_handle(
        payload, expected=GameOverServerMessage
    )

    assert decoded.MessageType is ServerMessageType.GAMEOVER
    assert decoded.MessageBody.ActualCoverage == 12
    assert decoded.MessageBody.TestsCount == 3
    assert decoded.MessageBody.ErrorsCount == 1
