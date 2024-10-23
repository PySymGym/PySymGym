import csv
import pytest
from ml.training.statistics import get_svms_statistics
from ml.training.dataset import Result
from pathlib import Path
from tests.utils import read_configs
import yaml
from common.classes import GameFailed, Map2Result, GameResult
from run_training import get_maps
from random import choice
from common.config import ValidationConfig
from unittest import mock
import shutil


class TrainingDatasetMock:
    def __init__(self, mock_maps_results: dict[str, Result]) -> None:
        self.maps_results = mock_maps_results


class TestSVMsStatistics:
    tmp_dir = Path("tests/tmp")
    test_csv_file_path = Path(tmp_dir / "test_statistics.csv")
    mock.patch("paths.CURRENT_TABLE_PATH", test_csv_file_path)

    def create_statistics_file_with_header(self, maps):
        self.tmp_dir.mkdir(exist_ok=True)
        with open(self.test_csv_file_path, "w") as statistics_file:
            statistics_writer = csv.DictWriter(
                statistics_file,
                sorted([game_map2svm.GameMap.MapName for game_map2svm in maps]),
            )
            statistics_writer.writeheader()

    def remove_tmp(self):
        shutil.rmtree("tests/tmp")

    def get_args(self, validation_config):
        with open(validation_config) as file:
            val_config = ValidationConfig(**yaml.safe_load(file)).validation

        dataset = TrainingDatasetMock(
            mock_maps_results={
                "Map1": Result(100, -4, -9028, 5),
                "Map2": Result(0, -1, -8, 0),
            }
        )
        maps = get_maps(val_config)
        return val_config, dataset, maps

    @pytest.mark.parametrize(
        "validation_config", read_configs("tests/resources/svms_validation_configs")
    )
    def test_single_epoch_with_successful_results(self, validation_config):
        metrics = {
            "average_coverage": 100.0,
            "average_coverage_for_VSharp": 100.0,
            "average_coverage_for_usvm": 100.0,
            "average_dataset_state_coverage": 50.0,
            "failed_maps_number_for_VSharp": 0,
            "failed_maps_number_for_usvm": 0,
        }
        val_config, dataset, maps = self.get_args(validation_config)
        self.create_statistics_file_with_header(maps)
        try:
            results = [
                Map2Result(game_map2svm, GameResult(1000, 2, 3, 100))
                for game_map2svm in maps
            ]
            assert get_svms_statistics(results, val_config, dataset) == (100.0, metrics)
            self.remove_tmp()
        except Exception as exc:
            self.remove_tmp()
            raise exc

    @pytest.mark.parametrize(
        "validation_config", read_configs("tests/resources/svms_validation_configs")
    )
    def test_csv_updating_with_two_epochs_with_removed_maps(self, validation_config):
        val_config, dataset, maps = self.get_args(validation_config)
        self.create_statistics_file_with_header(maps)
        try:
            results = [
                Map2Result(
                    game_map2svm, choice([GameResult(1000, 2, 3, 100), GameFailed(reason=Exception)])
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
            self.remove_tmp()
        except Exception as exc:
            self.remove_tmp()
            raise exc