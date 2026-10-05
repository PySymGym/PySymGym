"""Unit tests for the connection error hierarchy."""

import inspect

import pytest

from connection.errors_connection import (
    ConnectionLostError,
    GameInterruptedError,
    ProcessStoppedError,
)

pytestmark = pytest.mark.unit


def test_game_interrupted_error_is_abstract() -> None:
    assert inspect.isabstract(GameInterruptedError)
    assert "desc" in GameInterruptedError.__abstractmethods__


@pytest.mark.parametrize(
    ("error_cls", "expected_desc"),
    [
        (ProcessStoppedError, "SVM's process unexpectedly stopped"),
        (ConnectionLostError, "Connection to SVM was lost"),
    ],
)
def test_concrete_errors_expose_description(
    error_cls: type[GameInterruptedError], expected_desc: str
) -> None:
    assert not inspect.isabstract(error_cls)
    error = error_cls()
    assert isinstance(error, GameInterruptedError)
    assert error.desc == expected_desc
