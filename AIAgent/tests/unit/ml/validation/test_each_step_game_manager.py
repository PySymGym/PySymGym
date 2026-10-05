"""Unit tests for ``EachStepGameManager`` with a fake predictor and connector.

The socket manager and ``Connector`` are patched, so the real step loop runs
against a scripted connector; graph inputs come from the shared
``gamestate_factory`` and are converted by the production
``convert_input_to_tensor`` path.
"""

from contextlib import contextmanager

import pytest
import torch
from common.classes import GameFailed, GameResult, Map2Result
from common.game import GameMap, GameState
from config import FeatureConfig
from connection.game_server_conn.connector import Connector
from ml.protocols import Predictor
from ml.validation.coverage.game_managers.each_step import (
    each_step_game_manager as esgm,
)
from ml.validation.coverage.game_managers.each_step.each_step_game_manager import (
    EachStepGameManager,
)

pytestmark = [pytest.mark.unit, pytest.mark.serial]


class FakePredictor(Predictor):
    def __init__(self, outputs: list[list[float]] | None = None) -> None:
        self._outputs = outputs or [[1.0, 0.0]]
        self._model = torch.nn.Linear(1, 1)
        self.steps = 0

    def name(self) -> str:
        return "fake-predictor"

    def model(self) -> torch.nn.Module:
        return self._model

    def predict(self, input, map_name=None):
        output = self._outputs[min(self.steps, len(self._outputs) - 1)]
        self.steps += 1
        return 0, output


class FakeConnector:
    def __init__(
        self,
        game_map: GameMap,
        states: list,
        gameover_after: int,
        gameover: Connector.GameOver,
        fail_with: BaseException | None = None,
    ) -> None:
        self.map = game_map
        self._states = list(states)
        self._gameover_after = gameover_after
        self._gameover = gameover
        self._fail_with = fail_with
        self._received = 0
        self.sent_steps: list[int] = []

    def recv_state_or_throw_gameover(self):
        if self._fail_with is not None:
            raise self._fail_with
        if self._received >= self._gameover_after:
            raise self._gameover
        self._received += 1
        return self._states.pop(0)

    def send_step(self, next_state_id: int, predicted_usefullness: float) -> None:
        self.sent_steps.append(next_state_id)

    def recv_reward_or_throw_gameover(self):
        return object()


class FakeConnectorFactory:
    """Callable stand-in for ``Connector`` that still exposes ``GameOver``.

    ``_play_game_map_with_svm`` catches ``Connector.GameOver``, so the patched
    name must be an object with both a call and a ``GameOver`` attribute.
    """

    GameOver = Connector.GameOver

    def __init__(self, connector: FakeConnector) -> None:
        self._connector = connector

    def __call__(self, ws, game_map) -> FakeConnector:
        return self._connector


class FakeSaveFeature:
    def __init__(self) -> None:
        self.saved: list[tuple] = []

    def save_model(self, model, with_name: str) -> None:
        self.saved.append((model, with_name))


def _patch_connector(monkeypatch: pytest.MonkeyPatch, fake: FakeConnector) -> None:
    monkeypatch.setattr(esgm, "Connector", FakeConnectorFactory(fake))


def _manager(fake_namespace, predictor: FakePredictor) -> EachStepGameManager:
    return EachStepGameManager(predictor, fake_namespace)


def _gameover(coverage: int, tests: int, steps: int, errors: int) -> Connector.GameOver:
    return Connector.GameOver(
        actual_coverage=coverage,
        tests_count=tests,
        steps_count=steps,
        errors_count=errors,
    )


def test_play_game_map_with_svm_returns_result_and_steps(
    fake_namespace,
    game_map2svm_factory,
    gamestate_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_map2svm = game_map2svm_factory(steps_to_play=2)
    first = gamestate_factory()
    second = gamestate_factory()
    # update_game_state concatenates path-condition vertices, so the delta
    # carries none (the first snapshot already has them) to avoid duplicates.
    delta = GameState(
        GraphVertices=second.GraphVertices,
        States=second.States,
        PathConditionVertices=[],
        Map=second.Map,
    )
    fake = FakeConnector(
        game_map2svm.GameMap,
        states=[first, delta],
        gameover_after=2,
        gameover=_gameover(100, 1, 2, 0),
    )
    _patch_connector(monkeypatch, fake)
    manager = _manager(fake_namespace, FakePredictor())

    result, duration = manager._play_game_map_with_svm(game_map2svm, object())

    assert isinstance(result, GameResult)
    assert result == GameResult(
        steps_count=2,
        tests_count=1,
        errors_count=0,
        actual_coverage_percent=100,
    )
    assert duration >= 0
    steps = manager.get_game_steps(game_map2svm.GameMap)
    assert steps is not None
    assert len(steps) == 2


def test_play_game_map_with_svm_immediate_gameover(
    fake_namespace, game_map2svm_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    game_map2svm = game_map2svm_factory(steps_to_play=5)
    fake = FakeConnector(
        game_map2svm.GameMap,
        states=[],
        gameover_after=0,
        gameover=_gameover(100, 0, 0, 0),
    )
    _patch_connector(monkeypatch, fake)
    manager = _manager(fake_namespace, FakePredictor())

    result, _ = manager._play_game_map_with_svm(game_map2svm, object())

    assert result == GameResult(
        steps_count=5,
        tests_count=0,
        errors_count=0,
        actual_coverage_percent=0,
    )


def test_play_game_map_returns_map_result(
    fake_namespace,
    game_map2svm_factory,
    gamestate_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_map2svm = game_map2svm_factory(steps_to_play=1)
    fake = FakeConnector(
        game_map2svm.GameMap,
        states=[gamestate_factory()],
        gameover_after=1,
        gameover=_gameover(100, 1, 1, 0),
    )
    _patch_connector(monkeypatch, fake)

    @contextmanager
    def fake_socket_manager(svm_info):
        yield object()

    monkeypatch.setattr(esgm, "game_server_socket_manager", fake_socket_manager)
    save_feature = FakeSaveFeature()
    monkeypatch.setattr(FeatureConfig, "SAVE_IF_FAIL_OR_TIMEOUT", save_feature)
    manager = _manager(fake_namespace, FakePredictor())

    result = manager._play_game_map(game_map2svm)

    assert isinstance(result, Map2Result)
    assert isinstance(result.game_result, GameResult)
    assert save_feature.saved == []


def test_play_game_map_saves_model_on_failure(
    fake_namespace, game_map2svm_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    game_map2svm = game_map2svm_factory(steps_to_play=1)
    fake = FakeConnector(
        game_map2svm.GameMap,
        states=[],
        gameover_after=0,
        gameover=_gameover(0, 0, 0, 0),
        fail_with=RuntimeError("predictor exploded"),
    )
    _patch_connector(monkeypatch, fake)

    @contextmanager
    def fake_socket_manager(svm_info):
        yield object()

    monkeypatch.setattr(esgm, "game_server_socket_manager", fake_socket_manager)
    save_feature = FakeSaveFeature()
    monkeypatch.setattr(FeatureConfig, "SAVE_IF_FAIL_OR_TIMEOUT", save_feature)
    predictor = FakePredictor()
    manager = _manager(fake_namespace, predictor)

    result = manager._play_game_map(game_map2svm)

    assert isinstance(result.game_result, GameFailed)
    assert save_feature.saved == [(predictor.model(), "fake-predictor")]
