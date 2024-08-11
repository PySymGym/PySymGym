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


@dataclass
class StatsWithTable:
    avg: float
    df: pd.DataFrame


class Status:
    def __init__(self):
        self.epoch: EpochNumber = 0
        self.failed_maps: list[GameMap2SVM] = []
        self.count_of_failed_maps: int = 0

    def __str__(self) -> str:
        result = (
            f"count of failed maps={self.count_of_failed_maps}, on epoch = {self.epoch}"
        )
        return result


class StatisticsCollector:
    def __init__(
        self,
        file: Path,
    ):
        self._file = file
        self.lock = multiprocessing.Lock()

        self._svms_info: dict[SVMName, StatsWithTable] = {}
        self._svms_status: dict[SVMName, Status] = {}

    @staticmethod
    def avg_by_attr(results, path_to_coverage: str) -> int:
        coverage = np.average(
            list(map(lambda result: getattr(result, path_to_coverage), results))
        )
        return coverage

    def update_file(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            self.__update_file()
            return res

        return wrapper

    def fail(self, game_map: GameMap2SVM):
        svm_name = game_map.SVMInfo.name
        with self.lock:
            svm_status: Status = self._svms_status.get(svm_name, Status())
            svm_status.failed_maps.append(game_map)
            svm_status.count_of_failed_maps += 1
            self._svms_status[svm_name] = svm_status

    def __clear_failed_maps(self):
        for svm_status in self._svms_status.values():
            svm_status.failed_maps.clear()

    def get_failed_maps(self) -> list[GameMap2SVM]:
        """
        Returns failed maps.

        NB: The list of failed maps is cleared after each request.
        """
        total_failed_maps: list[GameMap2SVM] = []
        for svm_status in self._svms_status.values():
            total_failed_maps.extend(svm_status.failed_maps)
        self.__clear_failed_maps()
        return total_failed_maps

    @update_file
    def update_results(
        self,
        map2results_list: list[Map2Result],
    ):
        def generate_dict_with_svms_result() -> dict[SVMName, list[Map2Result]]:
            svms_and_map2results_lists: dict[SVMName, list[Map2Result]] = dict()
            for map2result in map2results_list:
                svm_name = map2result.map.SVMInfo.name
                map2results_list_of_svm = svms_and_map2results_lists.get(svm_name, [])
                map2results_list_of_svm.append(map2result)
                svms_and_map2results_lists[svm_name] = map2results_list_of_svm
            return svms_and_map2results_lists

        def generate_svms_info(svms_and_map2results: dict[SVMName, list[Map2Result]]):
            svms_info: dict[SVMName, list[StatsWithTable]] = dict()
            for svm_name, map2results_list in svms_and_map2results.items():
                svms_info[svm_name] = StatsWithTable(
                    StatisticsCollector.avg_by_attr(
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
            return svms_info

        svms_and_map2results_lists = generate_dict_with_svms_result()
        svms_info = generate_svms_info(svms_and_map2results_lists)

        self._svms_info = sort_dict(svms_info)

    def __get_results(self) -> str:
        svms_info = self._svms_info.items()
        _, svms_stats_with_table = list(zip(*svms_info))

        avg_coverage = StatisticsCollector.avg_by_attr(svms_stats_with_table, "avg")
        df_concat = pd.concat(
            list(
                map(lambda stats_with_table: stats_with_table.df, svms_stats_with_table)
            ),
            axis=1,
        )

        results = (
            f"Average coverage: {str(avg_coverage)}\n"
            + "".join(
                list(
                    map(
                        lambda pair: f"Average coverage of {pair[0]} = {pair[1].avg}\n",
                        svms_info,
                    )
                )
            )
            + df_concat.to_markdown(tablefmt="psql")
        )
        return results

    def __update_file(self):
        results = self.__get_results()
        with open(self._file, "w") as f:
            f.write(results)


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
