"""Unit tests for the connector exceptions and GameOver parsing.

These exercise the pure parts of ``Connector`` without opening a websocket:
``_raise_if_gameover`` only reads an already-received message string.
"""

import json
from types import SimpleNamespace
from typing import cast

import pytest

from connection.game_server_conn.connector import Connector
from connection.game_server_conn.messages import PROTOCOL_VERSION

pytestmark = pytest.mark.unit


def _raise_if_gameover(fake_self: SimpleNamespace, payload: str) -> str | bytes:
    # Exercises the unbound method with a duck-typed self: it only reads the
    # attributes the fakes expose (game_is_over, _disable_message_checks,
    # _gameover) and never touches the websocket.
    return Connector._raise_if_gameover(cast(Connector, fake_self), payload)


def test_game_over_exception_stores_counts() -> None:
    exc = Connector.GameOver(
        actual_coverage=10, tests_count=2, steps_count=3, errors_count=1
    )
    assert exc.actual_coverage == 10
    assert exc.tests_count == 2
    assert exc.steps_count == 3
    assert exc.errors_count == 1


def test_raise_if_gameover_parses_gameover_message() -> None:
    payload = json.dumps(
        {
            "Version": PROTOCOL_VERSION,
            "MessageType": "GameOver",
            "MessageBody": {
                "ActualCoverage": 55,
                "TestsCount": 6,
                "StepsCount": 7,
                "ErrorsCount": 2,
            },
        }
    )
    fake_self = SimpleNamespace(game_is_over=False, _disable_message_checks=True)

    with pytest.raises(Connector.GameOver) as exc_info:
        _raise_if_gameover(fake_self, payload)

    assert fake_self.game_is_over is True
    assert exc_info.value.actual_coverage == 55
    assert exc_info.value.tests_count == 6
    assert exc_info.value.steps_count == 7
    assert exc_info.value.errors_count == 2


def test_raise_if_gameover_passes_through_other_messages() -> None:
    payload = json.dumps(
        {"Version": PROTOCOL_VERSION, "MessageType": "ReadyForNextStep"}
    )
    fake_self = SimpleNamespace(game_is_over=False, _disable_message_checks=False)

    assert _raise_if_gameover(fake_self, payload) == payload


def test_raise_if_gameover_already_over_raises_gameover() -> None:
    stored = Connector.GameOver(
        actual_coverage=55, tests_count=6, steps_count=7, errors_count=2
    )
    fake_self = SimpleNamespace(game_is_over=True, _gameover=stored)

    with pytest.raises(Connector.GameOver) as exc_info:
        _raise_if_gameover(fake_self, "{}")

    # The guard re-raises the stored exception with its original counts.
    assert exc_info.value is stored
    assert exc_info.value.actual_coverage == 55
    assert exc_info.value.tests_count == 6
    assert exc_info.value.steps_count == 7
    assert exc_info.value.errors_count == 2


def test_raise_if_gameover_strict_validation_reads_steps_count() -> None:
    payload = json.dumps(
        {
            "Version": PROTOCOL_VERSION,
            "MessageType": "GameOver",
            "MessageBody": {
                "ActualCoverage": 55,
                "TestsCount": 6,
                "StepsCount": 7,
                "ErrorsCount": 2,
            },
        }
    )
    fake_self = SimpleNamespace(game_is_over=False, _disable_message_checks=False)

    with pytest.raises(Connector.GameOver) as exc_info:
        _raise_if_gameover(fake_self, payload)

    assert fake_self.game_is_over is True
    assert exc_info.value.actual_coverage == 55
    assert exc_info.value.tests_count == 6
    assert exc_info.value.steps_count == 7
    assert exc_info.value.errors_count == 2
