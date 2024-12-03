import csv
import pytest
from ml.validation.statistics import (
    get_svms_statistics,
    AVERAGE_COVERAGE,
    AVERAGE_COVERAGE_OF_DATASET_STATE,
    SVM_FAILED_MAPS_NUM_PREFIX,
    SVM_AVERAGE_COVERAGE_PREFIX,
)
from ml.dataset import Result
from pathlib import Path
from tests.utils import read_configs
import yaml
from common.classes import GameFailed, Map2Result, GameResult
from run_training import get_maps
from random import choice
from common.config import ValidationConfig
import shutil
import paths


class TrainingDatasetMock:
    def __init__(self, mock_maps_results: dict[str, Result]) -> None:
        self.maps_results = mock_maps_results


class TestSVMsStatistics:
    def create_statistics_file_with_header(self, maps):
        self.tmp_dir.mkdir(exist_ok=True)
        with open(self.test_csv_file_path, "w") as statistics_file:
            statistics_writer = csv.DictWriter(
                statistics_file,
                sorted([game_map2svm.GameMap.MapName for game_map2svm in maps]),
            )
            statistics_writer.writeheader()

    def remove_tmp(self):
        shutil.rmtree(self.tmp_dir)

    @pytest.fixture(autouse=True)
    def mock_variables_and_create_tmp(self, monkeypatch):
        self.tmp_dir = Path("./tests/tmp")
        self.test_csv_file_path = Path(self.tmp_dir / "test_statistics.csv")
        monkeypatch.setattr(paths, "CURRENT_TABLE_PATH", self.test_csv_file_path)
        yield
        self.remove_tmp()

    @pytest.fixture(params=read_configs("tests/resources/svms_validation_configs"))
    def get_args(self, request):
        with open(request.param) as file:
            val_config = ValidationConfig(**yaml.safe_load(file)).validation

        dataset = TrainingDatasetMock(
            mock_maps_results={
                "Map1": Result(100, -4, -9028, 5),
                "Map2": Result(0, -1, -8, 0),
            }
        )
        maps = get_maps(val_config)
        return val_config, dataset, maps

    def test_single_epoch_with_successful_results(self, get_args):
        val_config, dataset, maps = get_args
        metrics = dict()
        metrics[AVERAGE_COVERAGE] = 100.0
        metrics[AVERAGE_COVERAGE_OF_DATASET_STATE] = 50.0
        for platform in val_config.platforms_config:
            for svm_info in platform.svms_info:
                metrics[SVM_FAILED_MAPS_NUM_PREFIX + svm_info.name] = 0
                metrics[SVM_AVERAGE_COVERAGE_PREFIX + svm_info.name] = 100.0
        self.create_statistics_file_with_header(maps)
        results = [
            Map2Result(game_map2svm, GameResult(1000, 2, 3, 100))
            for game_map2svm in maps
        ]

        assert get_svms_statistics(results, val_config, dataset) == metrics

    def test_csv_updating_with_two_epochs_with_removed_maps(self, get_args):
        val_config, dataset, maps = get_args
        self.create_statistics_file_with_header(maps)
        results = [
            Map2Result(
                game_map2svm,
                choice([GameResult(1000, 2, 3, 100), GameFailed(reason=Exception)]),
            )
            for game_map2svm in maps
        ]
        get_svms_statistics(results, val_config, dataset)
        results = results[0 : int(len(results) * 0.5)]
        get_svms_statistics(results, val_config, dataset)
        with open(self.test_csv_file_path) as f:
            reader = iter(csv.reader(f))
            header_len = len(next(reader))
            for row in reader:
                assert header_len == len(row)
