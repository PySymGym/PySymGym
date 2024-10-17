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
    with open(CURRENT_TABLE_PATH, "w") as statistics_file:
        maps_results = dict(
            list(
                map(
                    lambda map2result: (
                        map2result.map.GameMap.MapName,
                        str(map2result.game_result),
                        results,
                    )
                )
            )
        )
        statistics_writer = csv.DictWriter(statistics_file, sorted(maps_results.keys()))
        statistics_writer.writerow(maps_results)

    failed_maps = filter(
        lambda map2result: isinstance(map2result.game_result, GameFailed), results
    )
    successful_maps = filter(
        lambda map2result: not isinstance(map2result.game_result, GameFailed), results
    )
    average_result = avg_by_attr(
        list(map(lambda map2result: map2result.game_result, successful_maps)),
        "actual_coverage_percent",
    )
    metrics = {
        "average_dataset_state_coverage": avg_by_attr(
            dataset.maps_results.values(), "coverage_percent"
        ),
        "average_coverage": average_result,
    }

    for platform in validation_config.platforms_config:
        metrics[f"failed_maps_number for_{platform.name}"] = sum(
            1
            for map2result in failed_maps
            if map2result.map.SVMInfo.name == platform.name
        )
        metrics[f"average_coverage_for_{platform.name}"] = avg_by_attr(
            list(
                map(
                    lambda map2result: map2result.game_result
                    if map2result.map.SVMInfo.name == platform.name
                    else None,
                    successful_maps,
                )
            ),
            "actual_coverage_percent",
        )
    return metrics["average_coverage"], metrics
