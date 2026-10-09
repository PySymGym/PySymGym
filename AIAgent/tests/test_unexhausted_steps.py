"""Tests for the opt-in failure on "Not all steps exhausted" (issue #467).

The symbolic engine can end a game before all planned steps are played without
reaching 100% coverage; that is an engine defect, not a model quality issue.
With ``fail_on_unexhausted_steps`` set in the validation config such a map must
come back as ``GameFailed`` so that ``fail_immediately`` can fail the run (and
CI) instead of the condition being swallowed by a warning.
"""

import contextlib
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import pytest
from common.classes import GameFailed, GameResult
from common.config.validation_config import (
    SVMValidationSendEachStep,
    SVMValidationSendModel,
    ValidationConfig,
)
from common.game import GameMap, GameMap2SVM
from common.validation_coverage.svm_info import SVMInfo
from connection.game_server_conn.connector import Connector
from ml.validation.coverage.game_managers.each_step import each_step_game_manager
from ml.validation.coverage.game_managers.model import process_game_manager
from ml.validation.coverage.game_managers.utils import unexhausted_steps_failure
from ml.validation.coverage.validate_coverage import ValidationCoverage


def svms_validation_mode(val_type: str, **extra) -> dict:
    return {
        "val_type": val_type,
        "process_count": 1,
        "PlatformsConfig": [
            {
                "name": "dotnet",
                "DatasetConfigs": [
                    {
                        "dataset_base_path": "/tmp/maps",
                        "dataset_description": "/tmp/maps/dataset.json",
                    }
                ],
                "SVMSInfo": [
                    {
                        "name": "VSharp",
                        "launch_command": "dotnet runner.dll --mapname {MapName}",
                        "server_working_dir": "/tmp/server",
                        "min_port": 35100,
                        "max_port": 35150,
                    }
                ],
            }
        ],
        **extra,
    }


class TestConfigFlag:
    @pytest.mark.parametrize(
        ("val_type", "expected_cls"),
        [
            ("svms_model", SVMValidationSendModel),
            ("svms_each_step", SVMValidationSendEachStep),
        ],
    )
    def test_default_is_false(self, val_type, expected_cls):
        mode = ValidationConfig(
            validation_mode=svms_validation_mode(val_type)
        ).validation_mode
        assert isinstance(mode, expected_cls)
        assert mode.fail_on_unexhausted_steps is False

    @pytest.mark.parametrize("val_type", ["svms_model", "svms_each_step"])
    def test_explicit_true(self, val_type):
        mode = ValidationConfig(
            validation_mode=svms_validation_mode(
                val_type, fail_on_unexhausted_steps=True
            )
        ).validation_mode
        assert mode.fail_on_unexhausted_steps is True


class TestUnexhaustedStepsFailure:
    def test_reason_names_map_and_numbers(self):
        failure = unexhausted_steps_failure("TestMap", 67, 200, 28.0)
        assert isinstance(failure, GameFailed)
        assert "TestMap" in failure.reason
        assert "67 of 200" in failure.reason
        assert "28.00%" in failure.reason

    def test_reason_supports_negative_steps(self):
        failure = unexhausted_steps_failure("TestMap", -21, 50, 30.0)
        assert "-21 of 50" in failure.reason


def make_game_map(steps_to_play=200, steps_to_start=0, map_name="TestMap") -> GameMap:
    return GameMap(
        StepsToPlay=steps_to_play,
        StepsToStart=steps_to_start,
        AssemblyFullName="Test.dll",
        NameOfObjectToCover="TestMethod",
        DefaultSearcher="BFS",
        MapName=map_name,
    )


def make_svm_info() -> SVMInfo:
    return SVMInfo(
        name="VSharp",
        launch_command="dotnet runner.dll --mapname {MapName}",
        server_working_dir="/tmp/server",
        min_port=35100,
        max_port=35150,
    )


class FakeNamespace:
    shared_lock = None
    is_prepared = SimpleNamespace(value=False)


def make_model_manager(
    tmp_path, monkeypatch, flag
) -> process_game_manager.ModelGameManager:
    monkeypatch.setattr(process_game_manager, "SVMS_OUTPUT_PATH", tmp_path)
    return process_game_manager.ModelGameManager(
        FakeNamespace(), None, fail_on_unexhausted_steps=flag
    )


def write_result(tmp_path, map_name, coverage, tests, steps, errors):
    output_dir = tmp_path / map_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{map_name}result").write_text(
        f"{coverage} {tests} {steps} {errors}"
    )


class TestModelGameManagerUnexhaustedSteps:
    def play(
        self,
        tmp_path,
        monkeypatch,
        flag,
        coverage,
        steps,
        steps_to_play=200,
        steps_to_start=0,
    ):
        manager = make_model_manager(tmp_path, monkeypatch, flag)
        write_result(tmp_path, "TestMap", coverage, 3, steps, 2)
        game_map2svm = GameMap2SVM(
            make_game_map(steps_to_play, steps_to_start), make_svm_info()
        )
        proc = SimpleNamespace(poll=lambda: 0)
        return manager._get_result(game_map2svm, proc)

    def test_flag_off_keeps_warning_behavior(self, tmp_path, monkeypatch):
        result = self.play(tmp_path, monkeypatch, False, coverage=28, steps=67)
        assert isinstance(result, GameResult)
        assert result.steps_count == 67

    def test_flag_on_fails_unexhausted_game(self, tmp_path, monkeypatch):
        result = self.play(tmp_path, monkeypatch, True, coverage=28, steps=67)
        assert isinstance(result, GameFailed)
        assert "TestMap" in result.reason
        assert "67 of 200" in result.reason

    def test_flag_on_subtracts_steps_to_start(self, tmp_path, monkeypatch):
        result = self.play(
            tmp_path, monkeypatch, True, coverage=30, steps=75, steps_to_start=50
        )
        assert isinstance(result, GameFailed)
        assert "25 of 200" in result.reason

    def test_flag_on_full_coverage_is_not_a_failure(self, tmp_path, monkeypatch):
        result = self.play(tmp_path, monkeypatch, True, coverage=100, steps=67)
        assert isinstance(result, GameResult)

    def test_flag_on_all_steps_exhausted_is_not_a_failure(self, tmp_path, monkeypatch):
        result = self.play(tmp_path, monkeypatch, True, coverage=55, steps=200)
        assert isinstance(result, GameResult)


class FakePredictor:
    def name(self):
        return "fake"

    def predict(self, game_state):
        return 0, 0.5

    def model(self):
        return None


def make_fake_connector(coverage, steps):
    """A Connector that sends one game state, then ends with a fixed GameOver."""

    class FakeConnector(Connector):
        def __init__(self, ws, game_map):
            self.map = game_map
            self._coverage = coverage
            self._steps = steps
            self._state_sent = False

        def recv_state_or_throw_gameover(self):
            if not self._state_sent:
                self._state_sent = True
                return SimpleNamespace(States=[SimpleNamespace(Id=0)])
            raise Connector.GameOver(
                actual_coverage=self._coverage,
                tests_count=1,
                steps_count=self._steps,
                errors_count=0,
            )

        def send_step(self, next_state_id, predicted_usefullness):
            pass

        def recv_reward_or_throw_gameover(self):
            return None

    return FakeConnector


def make_each_step_manager(monkeypatch, flag, coverage, steps):
    monkeypatch.setattr(
        each_step_game_manager, "Connector", make_fake_connector(coverage, steps)
    )
    monkeypatch.setattr(
        each_step_game_manager, "convert_input_to_tensor", lambda state: (dict(), {})
    )
    return each_step_game_manager.EachStepGameManager(
        FakePredictor(), FakeNamespace(), fail_on_unexhausted_steps=flag
    )


class TestEachStepGameManagerUnexhaustedSteps:
    def play(self, monkeypatch, flag, coverage, steps):
        manager = make_each_step_manager(monkeypatch, flag, coverage, steps)
        game_map2svm = GameMap2SVM(make_game_map(), make_svm_info())
        result, _ = manager._play_game_map_with_svm(game_map2svm, ws=None)
        return result

    def test_flag_off_keeps_fudged_result(self, monkeypatch):
        result = self.play(monkeypatch, False, coverage=30, steps=67)
        assert isinstance(result, GameResult)
        assert result.steps_count == 200
        assert result.actual_coverage_percent == 30

    def test_flag_on_fails_unexhausted_game(self, monkeypatch):
        result = self.play(monkeypatch, True, coverage=30, steps=67)
        assert isinstance(result, GameFailed)
        assert "TestMap" in result.reason
        assert "67 of 200" in result.reason

    def test_flag_on_fails_step_overrun(self, monkeypatch):
        result = self.play(monkeypatch, True, coverage=96, steps=201)
        assert isinstance(result, GameFailed)
        assert "201 of 200" in result.reason

    def test_flag_on_full_coverage_is_not_a_failure(self, monkeypatch):
        result = self.play(monkeypatch, True, coverage=100, steps=67)
        assert isinstance(result, GameResult)

    def test_flag_on_all_steps_exhausted_is_not_a_failure(self, monkeypatch):
        result = self.play(monkeypatch, True, coverage=55, steps=200)
        assert isinstance(result, GameResult)

    def test_play_game_map_wraps_failure_without_raising(self, monkeypatch):
        manager = make_each_step_manager(monkeypatch, True, coverage=30, steps=67)
        monkeypatch.setattr(
            each_step_game_manager,
            "game_server_socket_manager",
            lambda svm_info: contextlib.nullcontext(None),
        )
        game_map2svm = GameMap2SVM(make_game_map(), make_svm_info())
        result = manager._play_game_map(game_map2svm)
        assert isinstance(result.game_result, GameFailed)
        assert "TestMap" in result.game_result.reason


class TestGameManagerWiring:
    def test_sendmodel_manager_receives_flag(self):
        config = ValidationConfig(
            validation_mode=svms_validation_mode(
                "svms_model", fail_on_unexhausted_steps=True
            )
        ).validation_mode
        with mp.Manager() as sync_manager:
            manager = ValidationCoverage(None, None)._get_game_manager(
                config, sync_manager
            )
        assert manager._fail_on_unexhausted_steps is True

    def test_each_step_manager_receives_flag(self):
        config = ValidationConfig(
            validation_mode=svms_validation_mode(
                "svms_each_step", fail_on_unexhausted_steps=True
            )
        ).validation_mode
        with mp.Manager() as sync_manager:
            manager = ValidationCoverage(None, None)._get_game_manager(
                config, sync_manager
            )
        assert manager._fail_on_unexhausted_steps is True

    def test_sendmodel_manager_receives_kwargs_and_trial_dir(self):
        config = ValidationConfig(
            validation_mode=svms_validation_mode("svms_model")
        ).validation_mode
        model_kwargs = {"hidden_channels": 64}
        trial_directory = Path("/tmp/trials/7")
        with mp.Manager() as sync_manager:
            manager = ValidationCoverage(
                None, None, model_kwargs=model_kwargs, trial_dir=trial_directory
            )._get_game_manager(config, sync_manager)
        assert manager._model_kwargs == model_kwargs
        assert manager._trial_dir == trial_directory
        assert manager._path_to_model == trial_directory / "model.pth"
        assert manager._onnx_path == trial_directory / "model.onnx"

    def test_sendmodel_manager_falls_back_to_report_root(self):
        config = ValidationConfig(
            validation_mode=svms_validation_mode("svms_model")
        ).validation_mode
        with mp.Manager() as sync_manager:
            manager = ValidationCoverage(None, None)._get_game_manager(
                config, sync_manager
            )
        assert manager._path_to_model.name == "model.pth"
        assert manager._onnx_path.name == "model.onnx"
