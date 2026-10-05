"""Unit tests for ``ValidationCoverage`` with a fake manager and dataset.

No multiprocessing runs: the manager is a fake ``BaseGameManager`` and the
public flow patches ``mp.Manager``/``mp.Pool`` and ``tqdm`` so the orchestration
loop is exercised in-process.
"""

import threading
from types import SimpleNamespace

import pytest
import torch
from common.classes import GameFailed, GameResult, Map2Result
from common.config.validation_config import (
    CriterionValidation,
    SVMValidationSendEachStep,
    SVMValidationSendModel,
)
from ml.dataset import Result
from ml.validation.coverage import validate_coverage as vc
from ml.validation.coverage.game_managers.base_game_manager import BaseGameManager
from ml.validation.coverage.game_managers.each_step.each_step_game_manager import (
    EachStepGameManager,
)
from ml.validation.coverage.game_managers.model.process_game_manager import (
    ModelGameManager,
)
from ml.validation.coverage.validate_coverage import ValidationCoverage

pytestmark = [pytest.mark.unit, pytest.mark.serial]


class FakeGameManager(BaseGameManager):
    def __init__(self, result, steps=None, error: BaseException | None = None) -> None:
        self._result = result
        self._steps = steps
        self._error = error
        self.notified: list[tuple] = []
        self.deleted: list = []

    def _create_preparator(self):
        return None

    def _play_game_map(self, game_map2svm):
        if self._error is not None:
            raise self._error
        return self._result

    def play_game_map(self, game_map2svm):
        return self._play_game_map(game_map2svm)

    def get_game_steps(self, game_map):
        return self._steps

    def delete_game_artifacts(self, game_map):
        self.deleted.append(game_map)

    def notify_steps_requirement(self, game_map, required):
        self.notified.append((game_map, required))


class FakeDataset:
    def __init__(self, required: bool = True) -> None:
        self._required = required
        self.updates: list[tuple] = []

    def is_update_map_required(self, map_name, map_result):
        return self._required

    def update_map(self, map_name, map_result, steps):
        self.updates.append((map_name, map_result, steps))


class FakeSyncManager:
    def Namespace(self):
        return SimpleNamespace()

    def Lock(self):
        return threading.Lock()

    def Value(self, typecode, value):
        return SimpleNamespace(value=value)


def _coverage(manager: FakeGameManager, dataset) -> ValidationCoverage:
    coverage = ValidationCoverage(torch.nn.Linear(1, 1), dataset)
    coverage._game_manager = manager
    return coverage


def test_evaluate_game_map_requires_an_initialized_manager(
    game_map2svm_factory,
) -> None:
    coverage = ValidationCoverage(torch.nn.Linear(1, 1), None)

    with pytest.raises(RuntimeError, match="not been initialized"):
        coverage._evaluate_game_map(game_map2svm_factory())


def test_evaluate_game_map_updates_dataset_when_required(
    game_map2svm_factory,
) -> None:
    game_map2svm = game_map2svm_factory()
    result = Map2Result(
        game_map2svm,
        GameResult(
            steps_count=5, tests_count=2, errors_count=0, actual_coverage_percent=90
        ),
    )
    manager = FakeGameManager(result, steps=["step"])
    dataset = FakeDataset(required=True)
    coverage = _coverage(manager, dataset)

    returned = coverage._evaluate_game_map(game_map2svm)

    assert returned is result
    assert manager.notified == [(game_map2svm.GameMap, True)]
    assert dataset.updates == [
        (game_map2svm.GameMap.MapName, Result(90, -2, -5, 0), ["step"])
    ]
    assert manager.deleted == [game_map2svm.GameMap]


def test_evaluate_game_map_skips_update_when_not_required(
    game_map2svm_factory,
) -> None:
    game_map2svm = game_map2svm_factory()
    result = Map2Result(game_map2svm, GameResult(5, 2, 0, 90))
    manager = FakeGameManager(result, steps=["step"])
    dataset = FakeDataset(required=False)
    coverage = _coverage(manager, dataset)

    coverage._evaluate_game_map(game_map2svm)

    assert manager.notified == [(game_map2svm.GameMap, False)]
    assert dataset.updates == []
    assert manager.deleted == [game_map2svm.GameMap]


def test_evaluate_game_map_skips_update_when_steps_missing(
    game_map2svm_factory,
) -> None:
    game_map2svm = game_map2svm_factory()
    result = Map2Result(game_map2svm, GameResult(5, 2, 0, 90))
    manager = FakeGameManager(result, steps=None)
    dataset = FakeDataset(required=True)
    coverage = _coverage(manager, dataset)

    coverage._evaluate_game_map(game_map2svm)

    assert dataset.updates == []
    assert manager.deleted == [game_map2svm.GameMap]


def test_evaluate_game_map_without_dataset_just_deletes(
    game_map2svm_factory,
) -> None:
    game_map2svm = game_map2svm_factory()
    result = Map2Result(game_map2svm, GameResult(5, 2, 0, 90))
    manager = FakeGameManager(result)
    coverage = _coverage(manager, None)

    returned = coverage._evaluate_game_map(game_map2svm)

    assert returned is result
    assert manager.deleted == [game_map2svm.GameMap]


def test_evaluate_game_map_skips_dataset_for_game_failed(
    game_map2svm_factory,
) -> None:
    game_map2svm = game_map2svm_factory()
    result = Map2Result(game_map2svm, GameFailed("no game"))
    manager = FakeGameManager(result)
    dataset = FakeDataset(required=True)
    coverage = _coverage(manager, dataset)

    coverage._evaluate_game_map(game_map2svm)

    assert dataset.updates == []
    assert manager.deleted == [game_map2svm.GameMap]


def test_evaluate_game_map_returns_exception_and_deletes(
    game_map2svm_factory,
) -> None:
    game_map2svm = game_map2svm_factory()
    error = RuntimeError("play failed")
    manager = FakeGameManager(result=None, error=error)
    dataset = FakeDataset(required=True)
    coverage = _coverage(manager, dataset)

    returned = coverage._evaluate_game_map(game_map2svm)

    assert returned is error
    assert dataset.updates == []
    assert manager.deleted == [game_map2svm.GameMap]


def test_get_game_manager_returns_each_step_manager() -> None:
    coverage = ValidationCoverage(torch.nn.Linear(1, 1), None)
    config = SVMValidationSendEachStep(
        val_type="svms_each_step", PlatformsConfig=[], process_count=1
    )

    manager = coverage._get_game_manager(config, FakeSyncManager())

    assert isinstance(manager, EachStepGameManager)


def test_get_game_manager_returns_model_manager() -> None:
    coverage = ValidationCoverage(torch.nn.Linear(1, 1), None)
    config = SVMValidationSendModel(
        val_type="svms_model", PlatformsConfig=[], process_count=1
    )

    manager = coverage._get_game_manager(config, FakeSyncManager())

    assert isinstance(manager, ModelGameManager)


def test_get_game_manager_raises_for_other_modes() -> None:
    coverage = ValidationCoverage(torch.nn.Linear(1, 1), None)
    config = CriterionValidation(val_type="loss", batch_size=1)

    with pytest.raises(RuntimeError, match="no game manager suitable"):
        coverage._get_game_manager(config, FakeSyncManager())


class FakePool:
    def __init__(self, process_count: int) -> None:
        self.process_count = process_count

    def __enter__(self) -> "FakePool":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def imap_unordered(self, func, maps, chunksize):
        return [func(game_map) for game_map in maps]


class FakeManagerContext:
    def __enter__(self):
        return SimpleNamespace()

    def __exit__(self, *exc) -> bool:
        return False


def test_validate_coverage_collects_all_results(
    game_map2svm_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maps = [game_map2svm_factory("Method_0"), game_map2svm_factory("Method_1")]
    results = {
        id(game_map): Map2Result(game_map, GameResult(5, 1, 0, 100))
        for game_map in maps
    }
    manager = FakeGameManager(result=None)
    manager.play_game_map = lambda game_map2svm: results[id(game_map2svm)]
    coverage = ValidationCoverage(torch.nn.Linear(1, 1), None)
    config = SVMValidationSendEachStep(
        val_type="svms_each_step", PlatformsConfig=[], process_count=2
    )

    monkeypatch.setattr(
        coverage, "_get_game_manager", lambda config, sync_manager: manager
    )
    monkeypatch.setattr(vc.mp, "Manager", FakeManagerContext)
    monkeypatch.setattr(vc.mp, "Pool", FakePool)
    monkeypatch.setattr(vc.tqdm, "tqdm", lambda iterable, **kwargs: iterable)

    collected = coverage.validate_coverage(maps, config)

    assert {id(result) for result in collected} == {
        id(result) for result in results.values()
    }
    assert {result.map.GameMap.MapName for result in collected} == {
        "Method_0",
        "Method_1",
    }
