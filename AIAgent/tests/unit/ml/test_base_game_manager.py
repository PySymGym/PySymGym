"""Unit tests for the base game manager/preparator contract.

A concrete fake records the order of ``_prepare`` vs ``_play_game_map`` calls;
the namespace is the shared ``fake_namespace`` fixture (a real lock, no
multiprocessing manager). The shared failure/timeout helpers (#673, #616) are
tested on the same fake.
"""

import logging
from typing import cast

import pytest
from common.classes import GameFailed, Map2Result
from common.config.feature_config import FeatureConfig
from common.game import GameMap2SVM
from connection.errors_connection import ProcessStoppedError
from ml.validation.coverage.game_managers.base_game_manager import (
    BaseGameManager,
    BaseGamePreparator,
)

pytestmark = [pytest.mark.unit, pytest.mark.serial]


class OrderPreparator(BaseGamePreparator):
    def __init__(self, namespace, events: list[str]) -> None:
        super().__init__(namespace)
        self._events = events

    def _prepare(self) -> None:
        self._events.append("prepare")


class RecordingManager(BaseGameManager):
    mode = "test"

    def __init__(self, namespace, *, feature_config: FeatureConfig) -> None:
        self.events: list[str] = []
        super().__init__(namespace, feature_config=feature_config)

    def _create_preparator(self) -> BaseGamePreparator:
        return OrderPreparator(self._namespace, self.events)

    def _play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        self.events.append("play")
        return cast("Map2Result", game_map2svm)

    def get_game_steps(self, game_map):
        return None

    def delete_game_artifacts(self, game_map):
        pass

    def notify_steps_requirement(self, game_map, required):
        pass


def _manager(fake_namespace) -> RecordingManager:
    return RecordingManager(fake_namespace, feature_config=FeatureConfig())


def test_prepare_runs_once_and_marks_prepared(fake_namespace) -> None:
    events: list[str] = []
    preparator = OrderPreparator(fake_namespace, events)

    preparator.prepare()
    preparator.prepare()

    assert events == ["prepare"]
    assert fake_namespace.is_prepared.value is True


def test_play_game_map_prepares_before_playing(fake_namespace) -> None:
    manager = _manager(fake_namespace)
    sentinel = cast("GameMap2SVM", object())

    result = manager.play_game_map(sentinel)

    assert result is sentinel
    assert manager.events == ["prepare", "play"]


def test_prepare_is_not_repeated_across_games(fake_namespace) -> None:
    manager = _manager(fake_namespace)
    sentinel = cast("GameMap2SVM", object())

    manager.play_game_map(sentinel)
    manager.play_game_map(sentinel)

    assert manager.events == ["prepare", "play", "play"]


def test_feature_config_is_stored_on_the_manager(fake_namespace) -> None:
    feature_config = FeatureConfig()

    manager = RecordingManager(fake_namespace, feature_config=feature_config)

    assert manager._feature_config is feature_config


def test_failure_reason_carries_the_type_and_message() -> None:
    assert BaseGameManager._failure_reason(RuntimeError("boom")) == (
        "RuntimeError: boom"
    )


def test_failure_reason_prefers_desc_of_interrupted_errors() -> None:
    # GameInterruptedError subclasses are constructed without a message (their
    # str() is empty); the human-readable text lives in ``desc``.
    assert BaseGameManager._failure_reason(ProcessStoppedError()) == (
        "ProcessStoppedError: SVM's process unexpectedly stopped"
    )


def test_timeout_message_names_map_mode_budget_and_last_step(
    fake_namespace,
) -> None:
    manager = _manager(fake_namespace)

    message = manager._timeout_message(
        map_name="Method_0", steps_budget=10, elapsed=12.34, last_step=4
    )

    assert message == (
        "[test] map Method_0 timed out after 12.3s (step budget 10) at step 4/10"
    )


def test_timeout_message_omits_last_step_when_absent(fake_namespace) -> None:
    manager = _manager(fake_namespace)

    message = manager._timeout_message(
        map_name="Method_0", steps_budget=10, elapsed=12.34
    )

    assert message == "[test] map Method_0 timed out after 12.3s (step budget 10)"


def test_timeout_failure_logs_the_message_at_error_and_builds_the_failure(
    fake_namespace, caplog: pytest.LogCaptureFixture
) -> None:
    manager = _manager(fake_namespace)

    with caplog.at_level(logging.ERROR):
        failure = manager._timeout_failure(
            map_name="Method_0",
            steps_budget=10,
            elapsed=12.34,
            last_step=4,
        )

    assert isinstance(failure, GameFailed)
    assert failure.reason == (
        "[test] map Method_0 timed out after 12.3s (step budget 10) at step 4/10"
    )
    assert failure.reason in caplog.text
