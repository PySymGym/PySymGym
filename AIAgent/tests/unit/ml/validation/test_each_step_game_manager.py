"""Unit tests for ``EachStepGameManager`` with a fake predictor and connector.

The socket manager and ``Connector`` are patched, so the real step loop runs
against a scripted connector; graph inputs come from the shared
``gamestate_factory`` and are converted by the production
``convert_input_to_tensor`` path.
"""

from contextlib import contextmanager
import logging

import pytest
import torch
from common.classes import GameFailed, GameResult, Map2Result
from common.config.broker_config import BrokerConfig
from common.config.connector_config import GameServerConnectorConfig
from common.config.feature_config import FeatureConfig, SaveIfFailOrTimeout
from common.game import GameMap, GameState
from connection.errors_connection import ProcessStoppedError
from connection.game_server_conn.connector import Connector
from func_timeout import FunctionTimedOut
from ml.protocols import Predictor
from ml.validation.coverage.game_managers.each_step import (
    each_step_game_manager as esgm,
)
from ml.validation.coverage.game_managers.each_step.each_step_game_manager import (
    EachStepGamePreparator,
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
        # The real predictors return the model output as a tensor
        # ((num_states, 1)); the fake mirrors that shape.
        output = self._outputs[min(self.steps, len(self._outputs) - 1)]
        self.steps += 1
        return 0, torch.tensor(output)


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
        self.sent_usefulness: list[float] = []

    def recv_state_or_throw_gameover(self):
        if self._fail_with is not None:
            raise self._fail_with
        if self._received >= self._gameover_after:
            raise self._gameover
        self._received += 1
        return self._states.pop(0)

    def send_step(self, next_state_id: int, predicted_usefullness: float) -> None:
        self.sent_steps.append(next_state_id)
        self.sent_usefulness.append(predicted_usefullness)

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

    def __call__(self, ws, game_map, feature_config) -> FakeConnector:
        return self._connector


def _patch_connector(monkeypatch: pytest.MonkeyPatch, fake: FakeConnector) -> None:
    monkeypatch.setattr(esgm, "Connector", FakeConnectorFactory(fake))


def _feature_config(tmp_path) -> FeatureConfig:
    # The save path is redirected to tmp so the save-on-failure branch never
    # writes into the real report directory.
    return FeatureConfig(
        save_if_fail_or_timeout=SaveIfFailOrTimeout(
            save_path=tmp_path / "timed_out_agents"
        )
    )


def _patch_torch_save(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    saved: list[tuple] = []

    def fake_save(state, path):
        saved.append((state, path))

    monkeypatch.setattr(esgm.torch, "save", fake_save)
    return saved


def _manager(
    fake_namespace,
    predictor: FakePredictor,
    feature_config: FeatureConfig | None = None,
) -> EachStepGameManager:
    return EachStepGameManager(
        predictor,
        fake_namespace,
        feature_config=feature_config or FeatureConfig(),
        connector_config=GameServerConnectorConfig(),
        broker_config=BrokerConfig(),
    )


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
    assert fake.sent_steps == [0, 0]
    # The predicted usefulness is the model's predicted value for the chosen
    # (argmax) state (#671): the max element of the (num_states, 1) output.
    assert fake.sent_usefulness == [1.0, 1.0]
    steps = manager.get_game_steps(game_map2svm.GameMap)
    assert steps is not None
    assert len(steps) == 2


def test_play_game_map_with_svm_reports_model_played_steps(
    fake_namespace,
    game_map2svm_factory,
    gamestate_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The server reports the total trace steps (start + play); the metric is
    # the model-played count (total minus StepsToStart), identical to the
    # send-model mode. The agent plays 2 steps here, so the server total is
    # 2 (play) + 3 (start) = 5 and the reported steps_count is 5 - 3 = 2.
    game_map2svm = game_map2svm_factory(steps_to_play=5, steps_to_start=3)
    first = gamestate_factory()
    second = gamestate_factory()
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
        gameover=_gameover(100, 1, 5, 0),
    )
    _patch_connector(monkeypatch, fake)
    manager = _manager(fake_namespace, FakePredictor())

    result, _ = manager._play_game_map_with_svm(game_map2svm, object())

    assert isinstance(result, GameResult)
    assert result.steps_count == 2
    assert result.actual_coverage_percent == 100


def test_play_game_map_logs_progress_at_the_interval(
    fake_namespace,
    game_map2svm_factory,
    gamestate_factory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A stuck/slow map must be diagnosable from the log alone (#616): the
    # loop logs a heartbeat every PROGRESS_LOG_EVERY_N_STEPS steps with the
    # map name, the step count, and the elapsed time.
    game_map2svm = game_map2svm_factory(steps_to_play=2)
    first = gamestate_factory()
    second = gamestate_factory()
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
    # Log every step so the 2-step game produces a heartbeat per step.
    monkeypatch.setattr(esgm, "PROGRESS_LOG_EVERY_N_STEPS", 1)
    manager = _manager(fake_namespace, FakePredictor())

    with caplog.at_level(logging.INFO):
        manager._play_game_map_with_svm(game_map2svm, object())

    assert "Method_0: step 1/2" in caplog.text
    assert "Method_0: step 2/2" in caplog.text


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
    tmp_path,
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
    def fake_socket_manager(svm_info, connector_config, broker_config):
        yield object()

    monkeypatch.setattr(esgm, "game_server_socket_manager", fake_socket_manager)
    saved = _patch_torch_save(monkeypatch)
    manager = _manager(fake_namespace, FakePredictor(), _feature_config(tmp_path))

    result = manager._play_game_map(game_map2svm)

    assert isinstance(result, Map2Result)
    assert isinstance(result.game_result, GameResult)
    assert saved == []


def test_play_game_map_saves_model_on_failure(
    fake_namespace,
    game_map2svm_factory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
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
    def fake_socket_manager(svm_info, connector_config, broker_config):
        yield object()

    monkeypatch.setattr(esgm, "game_server_socket_manager", fake_socket_manager)
    saved = _patch_torch_save(monkeypatch)
    predictor = FakePredictor()
    manager = _manager(fake_namespace, predictor, _feature_config(tmp_path))

    result = manager._play_game_map(game_map2svm)

    assert isinstance(result.game_result, GameFailed)
    # The reason names the exception type and message, not the class (#673).
    assert result.game_result.reason == "RuntimeError: predictor exploded"
    # The failed model's state dict is saved under the feature config's
    # save path, named after the predictor with a timestamp suffix.
    assert len(saved) == 1
    state, path = saved[0]
    expected_state = predictor.model().state_dict()
    assert set(state) == set(expected_state)
    for name in expected_state:
        assert torch.equal(state[name], expected_state[name])
    assert path.parent == tmp_path / "timed_out_agents"
    assert path.name.startswith("fake-predictor_")
    assert path.name.endswith(".pt")


def test_play_game_map_timeout_saves_the_model(
    fake_namespace,
    game_map2svm_factory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path,
) -> None:
    game_map2svm = game_map2svm_factory(steps_to_play=1)

    @contextmanager
    def fake_socket_manager(svm_info, connector_config, broker_config):
        yield object()

    monkeypatch.setattr(esgm, "game_server_socket_manager", fake_socket_manager)

    # Simulates the wrapper raising on a real wall-clock timeout; the same
    # exception reaches the except branch below.
    def wall_clock_timeout(*args, **kwargs):
        raise FunctionTimedOut(timedOutAfter=1800)

    monkeypatch.setattr(esgm, "play_with_timeout", wall_clock_timeout)
    saved = _patch_torch_save(monkeypatch)
    predictor = FakePredictor()
    manager = _manager(fake_namespace, predictor, _feature_config(tmp_path))

    with caplog.at_level(logging.WARNING):
        result = manager._play_game_map(game_map2svm)

    assert isinstance(result.game_result, GameFailed)
    # The reason is the structured timeout message (#616), not the exception.
    assert result.game_result.reason.startswith(
        "[each-step] map Method_0 timed out after"
    )
    assert "(step budget 1)" in result.game_result.reason
    # The loop never completed a step, so the last observed step is 0.
    assert "at step 0/1" in result.game_result.reason
    assert len(saved) == 1
    assert result.game_result.reason in caplog.text


def test_play_game_map_inner_timeout_is_reported_as_a_timeout(
    fake_namespace,
    game_map2svm_factory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path,
) -> None:
    # A FunctionTimedOut raised from inside the wrapped function propagates
    # directly into the timeout branch (the worker re-raises it — unlike
    # func_timeout's swallow-and-return-None behaviour, #672), so the timeout
    # branch — not the generic exception branch — handles it and saves the
    # model.
    game_map2svm = game_map2svm_factory(steps_to_play=1)
    fake = FakeConnector(
        game_map2svm.GameMap,
        states=[],
        gameover_after=0,
        gameover=_gameover(0, 0, 0, 0),
        fail_with=FunctionTimedOut(timedOutAfter=1800),
    )
    _patch_connector(monkeypatch, fake)

    @contextmanager
    def fake_socket_manager(svm_info, connector_config, broker_config):
        yield object()

    monkeypatch.setattr(esgm, "game_server_socket_manager", fake_socket_manager)
    saved = _patch_torch_save(monkeypatch)
    predictor = FakePredictor()
    manager = _manager(fake_namespace, predictor, _feature_config(tmp_path))

    with caplog.at_level(logging.WARNING):
        result = manager._play_game_map(game_map2svm)

    assert isinstance(result.game_result, GameFailed)
    assert result.game_result.reason.startswith(
        "[each-step] map Method_0 timed out after"
    )
    assert "(step budget 1)" in result.game_result.reason
    assert "at step 0/1" in result.game_result.reason
    assert len(saved) == 1
    assert result.game_result.reason in caplog.text


def test_play_game_map_game_interrupted_does_not_save(
    fake_namespace,
    game_map2svm_factory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path,
) -> None:
    game_map2svm = game_map2svm_factory(steps_to_play=1)
    fake = FakeConnector(
        game_map2svm.GameMap,
        states=[],
        gameover_after=0,
        gameover=_gameover(0, 0, 0, 0),
        fail_with=ProcessStoppedError(),
    )
    _patch_connector(monkeypatch, fake)

    @contextmanager
    def fake_socket_manager(svm_info, connector_config, broker_config):
        yield object()

    monkeypatch.setattr(esgm, "game_server_socket_manager", fake_socket_manager)
    saved = _patch_torch_save(monkeypatch)
    manager = _manager(fake_namespace, FakePredictor(), _feature_config(tmp_path))

    with caplog.at_level(logging.WARNING):
        result = manager._play_game_map(game_map2svm)

    assert isinstance(result.game_result, GameFailed)
    # The reason carries the type and the ``desc`` message (#673).
    assert result.game_result.reason == (
        "ProcessStoppedError: SVM's process unexpectedly stopped"
    )
    # Unlike the timeout/failure branches, an interrupted game does not save.
    assert saved == []
    assert (
        "<fake-predictor> failed on map Method_0 with ProcessStoppedError: "
        "SVM's process unexpectedly stopped" in caplog.text
    )


def test_delete_game_artifacts_drops_the_stored_steps(
    fake_namespace, game_map_factory
) -> None:
    manager = _manager(fake_namespace, FakePredictor())
    game_map = game_map_factory("M")
    manager._game_states[str(game_map)] = ["step"]

    manager.delete_game_artifacts(game_map)

    assert manager._game_states == {}
    assert manager.get_game_steps(game_map) is None

    # A second call (absent key) is a no-op.
    manager.delete_game_artifacts(game_map)


def test_notify_steps_requirement_is_a_noop(fake_namespace, game_map_factory) -> None:
    manager = _manager(fake_namespace, FakePredictor())

    assert manager.notify_steps_requirement(game_map_factory("M"), True) is None


def test_preparator_prepare_marks_the_namespace(fake_namespace) -> None:
    manager = _manager(fake_namespace, FakePredictor())
    preparator = manager._create_preparator()
    assert isinstance(preparator, EachStepGamePreparator)

    preparator.prepare()

    assert fake_namespace.is_prepared.value is True
