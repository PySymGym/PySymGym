import csv
from ml.training.utils import avg_by_attr
from ml.training.dataset import TrainingDataset
from common.config import ValidationWithSVMs
import paths
from common.classes import GameFailed, Map2Result


def get_svms_statistics(
    results: list[Map2Result],
    validation_config: ValidationWithSVMs,
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
