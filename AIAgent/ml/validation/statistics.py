import csv

import paths
from common.classes import GameFailed, Map2Result
from common.config.validation_config import ValidationSVM
from ml.dataset import TrainingDataset
from ml.validation.statistics_utils import avg_by_attr

AVERAGE_COVERAGE_OF_DATASET_STATE = "average_dataset_state_coverage"
AVERAGE_COVERAGE = "average_coverage"
SVM_FAILED_MAPS_NUM_PREFIX = "failed_maps_number_for_"
SVM_AVERAGE_COVERAGE_PREFIX = "average_coverage_for_"
GAME_RESULT_COVERAGE_PERCENT_ATTR = "actual_coverage_percent"
RESULT_COVERAGE_PERCENT_FIELD_NAME = "coverage_percent"


def get_svms_statistics(
    results: list[Map2Result],
    validation_config: ValidationSVM,
    dataset: TrainingDataset,
):
    with open(paths.CURRENT_TABLE_PATH, "r") as statistics_file:
        header = next(iter(csv.reader(statistics_file)))
    maps_results = dict(
        [
            (map2result.map.GameMap.MapName, str(map2result.game_result))
            for map2result in results
        ]
    )
    results_to_write = dict(
        list(
            map(
                lambda map_name: (map_name, "DELETED")
                if map_name not in maps_results
                else (map_name, maps_results[map_name]),
                header,
            )
        )
    )
    with open(paths.CURRENT_TABLE_PATH, "a") as statistics_file:
        statistics_writer = csv.DictWriter(
            statistics_file, sorted(results_to_write.keys())
        )
        statistics_writer.writerow(results_to_write)
    failed_maps = [
        map2result
        for map2result in results
        if isinstance(map2result.game_result, GameFailed)
    ]
    successful_maps = [
        map2result
        for map2result in results
        if not isinstance(map2result.game_result, GameFailed)
    ]
    metrics = {
        AVERAGE_COVERAGE_OF_DATASET_STATE: avg_by_attr(
            dataset.maps_results.values(), RESULT_COVERAGE_PERCENT_FIELD_NAME
        ),
        AVERAGE_COVERAGE: avg_by_attr(
            [map2result.game_result for map2result in successful_maps],
            GAME_RESULT_COVERAGE_PERCENT_ATTR,
        ),
    }
    for platform in validation_config.platforms_config:
        for svm_info in platform.svms_info:
            metrics[SVM_FAILED_MAPS_NUM_PREFIX + svm_info.name] = sum(
                [
                    1
                    for map2result in failed_maps
                    if map2result.map.SVMInfo.name == svm_info.name
                ]
            )
            metrics[SVM_AVERAGE_COVERAGE_PREFIX + svm_info.name] = avg_by_attr(
                [
                    map2result.game_result
                    for map2result in successful_maps
                    if map2result.map.SVMInfo.name == svm_info.name
                ],
                GAME_RESULT_COVERAGE_PERCENT_ATTR,
            )
    return metrics
