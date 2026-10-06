"""Tests for the opt-in failure on "Not all steps exhausted" (issue #467).

The symbolic engine can end a game before all planned steps are played without
reaching 100% coverage; that is an engine defect, not a model quality issue.
With ``fail_on_unexhausted_steps`` set in the validation config such a map must
come back as ``GameFailed`` so that ``fail_immediately`` can fail the run (and
CI) instead of the condition being swallowed by a warning.
"""

import multiprocessing as mp
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
