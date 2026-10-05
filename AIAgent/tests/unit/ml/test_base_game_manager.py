"""Unit tests for the base game manager/preparator contract.

A concrete fake records the order of ``_prepare`` vs ``_play_game_map`` calls;
the namespace is the shared ``fake_namespace`` fixture (a real lock, no
multiprocessing manager).
"""

from typing import cast

import pytest
from common.classes import Map2Result
from common.game import GameMap2SVM
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
    def __init__(self, namespace) -> None:
        self.events: list[str] = []
        super().__init__(namespace)

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


def test_prepare_runs_once_and_marks_prepared(fake_namespace) -> None:
    events: list[str] = []
    preparator = OrderPreparator(fake_namespace, events)

    preparator.prepare()
    preparator.prepare()

    assert events == ["prepare"]
    assert fake_namespace.is_prepared.value is True


def test_play_game_map_prepares_before_playing(fake_namespace) -> None:
    manager = RecordingManager(fake_namespace)
    sentinel = cast("GameMap2SVM", object())

    result = manager.play_game_map(sentinel)

    assert result is sentinel
    assert manager.events == ["prepare", "play"]


def test_prepare_is_not_repeated_across_games(fake_namespace) -> None:
    manager = RecordingManager(fake_namespace)
    sentinel = cast("GameMap2SVM", object())

    manager.play_game_map(sentinel)
    manager.play_game_map(sentinel)

    assert manager.events == ["prepare", "play", "play"]
