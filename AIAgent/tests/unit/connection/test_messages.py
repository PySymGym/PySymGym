"""Unit tests for the game-server client/server messages."""

import dataclasses
import json
import logging

import pytest

from common.game import GameMap
from connection.errors_connection import ProtocolVersionMismatchError
from connection.game_server_conn import messages
from connection.game_server_conn.messages import (
    PROTOCOL_VERSION,
    ClientMessage,
    ClientMessageType,
    GameOverServerMessage,
    ServerMessage,
    ServerMessageType,
    StartMessageBody,
    StepMessageBody,
    check_protocol_version,
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
    message = ClientMessage(StartMessageBody.from_game_map(_game_map()))
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
    body = StartMessageBody.from_game_map(_game_map())
    assert StartMessageBody.from_json(body.to_json()) == body


def test_start_message_body_fields_mirror_game_map() -> None:
    # The sync test: the wire body mirrors GameMap field-for-field, so a
    # domain-model change cannot silently change the wire format (#582).
    body_fields = dataclasses.fields(StartMessageBody)
    map_fields = dataclasses.fields(GameMap)
    assert [f.name for f in body_fields] == [f.name for f in map_fields]
    assert [f.type for f in body_fields] == [f.type for f in map_fields]


def test_start_message_body_wire_json_equals_game_map_json() -> None:
    game_map = _game_map()
    assert StartMessageBody.from_game_map(game_map).to_json() == game_map.to_json()


@pytest.mark.parametrize("disabled", [True, False])
def test_from_json_handle_respects_message_checks(disabled: bool) -> None:
    payload = json.dumps({"Version": PROTOCOL_VERSION, "MessageType": "GameOver"})

    result = ServerMessage.from_json_handle(
        payload, expected=ServerMessage, disable_message_checks=disabled
    )

    if disabled:
        assert not isinstance(result, ServerMessage)
        assert result.MessageType == "GameOver"
    else:
        assert isinstance(result, ServerMessage)
        assert result.MessageType is ServerMessageType.GAMEOVER


def test_from_json_handle_raises_on_malformed_payload() -> None:
    payload = json.dumps({"Version": PROTOCOL_VERSION, "MessageType": "GameOver"})

    with pytest.raises(ServerMessage.DeserializationException):
        GameOverServerMessage.from_json_handle(
            payload,
            expected=GameOverServerMessage,
            disable_message_checks=False,
        )


def test_game_over_message_exposes_body() -> None:
    payload = json.dumps(
        {
            "Version": PROTOCOL_VERSION,
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
        payload, expected=GameOverServerMessage, disable_message_checks=False
    )

    assert decoded.MessageType is ServerMessageType.GAMEOVER
    assert decoded.MessageBody.ActualCoverage == 12
    assert decoded.MessageBody.TestsCount == 3
    assert decoded.MessageBody.StepsCount == 4
    assert decoded.MessageBody.ErrorsCount == 1


def test_check_protocol_version_accepts_the_current_version() -> None:
    assert check_protocol_version({"Version": PROTOCOL_VERSION}) is PROTOCOL_VERSION


def test_check_protocol_version_rejects_a_foreign_version() -> None:
    with pytest.raises(ProtocolVersionMismatchError) as exc_info:
        check_protocol_version({"Version": PROTOCOL_VERSION + 1})

    assert str(PROTOCOL_VERSION) in str(exc_info.value)
    assert str(PROTOCOL_VERSION + 1) in str(exc_info.value)


def test_check_protocol_version_ignores_non_dict_envelopes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(messages, "_legacy_version_notice_issued", False)

    with caplog.at_level(logging.WARNING):
        assert check_protocol_version([1, 2]) is None
        assert check_protocol_version("start") is None

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_check_protocol_version_warns_once_per_process_on_legacy(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(messages, "_legacy_version_notice_issued", False)

    with caplog.at_level(logging.WARNING):
        assert check_protocol_version({"MessageType": "GameOver"}) is None
        assert check_protocol_version({"MessageType": "GameOver"}) is None

    legacy_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "no Version field" in r.message
    ]
    assert len(legacy_warnings) == 1
    assert str(PROTOCOL_VERSION) in legacy_warnings[0].message


@pytest.mark.parametrize("disabled", [True, False])
def test_from_json_handle_decodes_a_v2_envelope(disabled: bool) -> None:
    payload = json.dumps({"Version": PROTOCOL_VERSION, "MessageType": "GameOver"})

    result = ServerMessage.from_json_handle(
        payload, expected=ServerMessage, disable_message_checks=disabled
    )

    if disabled:
        assert result.Version == PROTOCOL_VERSION
        assert result.MessageType == "GameOver"
    else:
        assert isinstance(result, ServerMessage)
        assert result.MessageType is ServerMessageType.GAMEOVER


@pytest.mark.parametrize("disabled", [True, False])
def test_from_json_handle_rejects_a_foreign_version_under_both_flags(
    disabled: bool,
) -> None:
    payload = json.dumps({"Version": PROTOCOL_VERSION + 1, "MessageType": "GameOver"})

    with pytest.raises(ProtocolVersionMismatchError):
        ServerMessage.from_json_handle(
            payload, expected=ServerMessage, disable_message_checks=disabled
        )


@pytest.mark.parametrize("disabled", [True, False])
def test_from_json_handle_wraps_invalid_json_under_both_flags(disabled: bool) -> None:
    with pytest.raises(ServerMessage.DeserializationException):
        ServerMessage.from_json_handle(
            "{not json", expected=ServerMessage, disable_message_checks=disabled
        )
