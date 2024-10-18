import csv
from ml.training.utils import avg_by_attr
from ml.training.dataset import TrainingDataset
from common.config import ValidationWithSVMs
from paths import CURRENT_TABLE_PATH
from common.game import GameMap2SVM
from common.classes import GameFailed


def get_svms_statistics(
    results: list[GameMap2SVM],
    validation_config: ValidationWithSVMs,
    dataset: TrainingDataset,
):
    with open(CURRENT_TABLE_PATH, "a") as statistics_file:
        maps_results = dict(
            [
                (map2result.map.GameMap.MapName, str(map2result.game_result))
                for map2result in results
            ]
        )
        statistics_writer = csv.DictWriter(statistics_file, sorted(maps_results.keys()))
        statistics_writer.writerow(maps_results)

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
        "average_dataset_state_coverage": avg_by_attr(
            dataset.maps_results.values(), "coverage_percent"
        ),
        "average_coverage": avg_by_attr(
            [map2result.game_result for map2result in successful_maps],
            "actual_coverage_percent",
        ),
    }
    for platform in validation_config.platforms_config:
        for svm_info in platform.svms_info:
            metrics[f"failed_maps_number_for_{svm_info.name}"] = sum(
                [
                    1
                    for map2result in failed_maps
                    if map2result.map.SVMInfo.name == svm_info.name
                ]
            )
            metrics[f"average_coverage_for_{svm_info.name}"] = avg_by_attr(
                [
                    map2result.game_result
                    for map2result in successful_maps
                    if map2result.map.SVMInfo.name == svm_info.name
                ],
                "actual_coverage_percent",
            )
    return metrics["average_coverage"], metrics
