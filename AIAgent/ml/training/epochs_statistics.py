from dataclasses import dataclass
from functools import wraps
import multiprocessing
from pathlib import Path
from typing import TypeAlias

import natsort
import numpy as np
import pandas as pd
from common.typealias import SVMName
from common.game import GameMap2SVM
from common.classes import Map2Result

EpochNumber: TypeAlias = int


def sort_dict(d):
    return dict(natsort.natsorted(d.items()))


def avg_by_attr(results, path_to_coverage: str) -> int:
    coverage = np.average(
        list(map(lambda result: getattr(result, path_to_coverage), results))
    )
    return coverage


@dataclass
class StatsWithTable:
    avg: float
    df: pd.DataFrame


class StatisticsWriter:
    def __init__(self, file):
        self._file = file
        self.lock = multiprocessing.Lock()

    def _succeed_maps_to_dict(
        self, succeed_maps: list[Map2Result]
    ) -> dict[SVMName, Map2Result]:
        res: dict[SVMName, Map2Result] = dict()
        svm_names = set(
            map(lambda map2result: map2result.map.SVMInfo.name, succeed_maps)
        )
        for svm_name in svm_names:
            res[svm_name] = list(
                filter(
                    lambda map2result: map2result.map.SVMInfo.name == svm_name,
                    succeed_maps,
                )
            )
        return res

    def _failed_maps_count_dict(
        self, failed_maps: list[GameMap2SVM]
    ) -> dict[SVMName, int]:
        res: dict[SVMName, int] = dict()
        svm_names = set(map(lambda map2svm: map2svm.SVMInfo.name, failed_maps))
        for svm_name in svm_names:
            res[svm_name] = len(
                list(
                    filter(
                        lambda map2svm: map2svm.SVMInfo.name == svm_name,
                        failed_maps,
                    )
                )
            )
        return res

    def _get_tables(
        self, succeed_maps_dict: dict[SVMName, list[Map2Result]]
    ) -> dict[SVMName, StatsWithTable]:
        svms_stats_dict: dict[SVMName, list[StatsWithTable]] = dict()
        for svm_name, map2results_list in succeed_maps_dict.items():
            svms_stats_dict[svm_name] = StatsWithTable(
                avg_by_attr(
                    list(
                        map(
                            lambda map2result: map2result.game_result,
                            map2results_list,
                        )
                    ),
                    "actual_coverage_percent",
                ),
                convert_to_df(map2results_list),
            )

        return sort_dict(svms_stats_dict)

    def _calc_avg(self, map2results_list: list[Map2Result]):
        return avg_by_attr(
            list(
                map(
                    lambda map2result: map2result.game_result,
                    map2results_list,
                )
            ),
            "actual_coverage_percent",
        )

    def _get_text_info(
        self,
        tables: dict[SVMName, StatsWithTable],
        avg_coverage: float,
        failed_maps_count_dict: dict[SVMName, int],
    ) -> str:
        svms_stats = tables.items()
        if svms_stats:
            _, svms_stats_with_table = list(zip(*svms_stats))
            table_markdown = pd.concat(
                list(
                    map(
                        lambda stats_with_table: stats_with_table.df,
                        svms_stats_with_table,
                    )
                ),
                axis=1,
            ).to_markdown(tablefmt="psql")
        else:
            svms_stats_with_table = []
            table_markdown = "Empty table"
        results = (
            f"Average coverage: {avg_coverage}\n"
            + "".join(
                list(
                    map(
                        lambda svm_name_and_stats_pair: f"Average coverage of {svm_name_and_stats_pair[0]} = {svm_name_and_stats_pair[1].avg}, count of failed maps = {failed_maps_count_dict.get(svm_name_and_stats_pair[0])}\n",
                        svms_stats,
                    )
                )
            )
            + table_markdown
        )
        return results

    def update_file(
        self,
        succeed_maps: list[Map2Result],
        failed_maps: list[GameMap2SVM],
    ):
        text_info = self._get_text_info(
            tables=self._get_tables(
                succeed_maps_dict=self._succeed_maps_to_dict(succeed_maps)
            ),
            avg_coverage=self._calc_avg(succeed_maps),
            failed_maps_count_dict=self._failed_maps_count_dict(failed_maps),
        )
        with self.lock:
            with open(self._file, "w") as f:
                f.write(text_info)


class StatisticsCollector:
    def __init__(
        self,
        file: Path,
    ):
        self.lock = multiprocessing.Lock()

        self._file_updater = StatisticsWriter(file)
        self._succeed_maps: list[Map2Result] = []
        self._failed_maps: list[GameMap2SVM] = []

    def update_file(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            self._file_updater.update_file(self._succeed_maps, self._failed_maps)
            return res

        return wrapper

    @update_file
    def fail(self, game_map: GameMap2SVM):
        self._failed_maps.append(game_map)

    @update_file
    def success(self, map2result: Map2Result):
        self._succeed_maps.append(map2result)

    def get_failed_maps(self) -> list[GameMap2SVM]:
        """
        Returns failed maps.
        """
        return self._failed_maps

    def get_succeed_map2results(self) -> list[Map2Result]:
        """
        Returns succeed maps.
        """
        return self._succeed_maps


def convert_to_df(map2result_list: list[Map2Result]) -> pd.DataFrame:
    maps = []
    results = []
    for map2result in map2result_list:
        _map = map2result.map
        map_name = _map.GameMap.MapName
        game_result_str = map2result.game_result.printable(verbose=True)
        maps.append(f"{_map.SVMInfo.name} : {map_name}")
        results.append(game_result_str)

    df = pd.DataFrame(results, columns=["Game result"], index=maps).T

    return df
