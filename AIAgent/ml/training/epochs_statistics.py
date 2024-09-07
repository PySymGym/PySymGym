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


class FailedMaps(list[GameMap2SVM]):
    lock = multiprocessing.Lock()

    def __str__(self) -> str:
        result = f"count of failed maps = {len(self)}"
        return result


class StatisticsCollector:
    def __init__(
        self,
        file: Path,
    ):
        self._file = file
        self.lock = multiprocessing.Lock()

        self._svms_stats_dict: dict[SVMName, StatsWithTable] = {}
        self._failed_maps_dict: dict[SVMName, FailedMaps] = {}

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
            failed_maps = self._failed_maps_dict.setdefault(svm_name, FailedMaps())
        with failed_maps.lock:
            failed_maps.append(game_map)

    def __clear_failed_maps(self):
        for failed_maps in self._failed_maps_dict.values():
            failed_maps.clear()

    def get_failed_maps(self) -> list[GameMap2SVM]:
        """
        Returns failed maps.

        NB: The list of failed maps is cleared after each request.
        """
        total_failed_maps: list[GameMap2SVM] = []
        for failed_maps in self._failed_maps_dict.values():
            total_failed_maps.extend(failed_maps)
        self.__clear_failed_maps()
        return total_failed_maps

    @update_file
    def update_results(
        self,
        map2results_list: list[Map2Result],
    ):
        def generate_svms_results_dict() -> dict[SVMName, list[Map2Result]]:
            map2results_dict: dict[SVMName, list[Map2Result]] = dict()
            for map2result in map2results_list:
                svm_name = map2result.map.SVMInfo.name
                map2results_list_of_svm = map2results_dict.get(svm_name, [])
                map2results_list_of_svm.append(map2result)
                map2results_dict[svm_name] = map2results_list_of_svm
            return map2results_dict

        def generate_svms_stats_dict(
            svms_and_map2results: dict[SVMName, list[Map2Result]]
        ):
            svms_stats_dict: dict[SVMName, list[StatsWithTable]] = dict()
            for svm_name, map2results_list in svms_and_map2results.items():
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
            return svms_stats_dict

        svms_and_map2results_lists = generate_svms_results_dict()
        svms_stats_dict = generate_svms_stats_dict(svms_and_map2results_lists)

        self._svms_stats_dict = sort_dict(svms_stats_dict)

    def __get_results(self) -> str:
        svms_stats = self._svms_stats_dict.items()
        _, svms_stats_with_table = list(zip(*svms_stats))

        avg_coverage = avg_by_attr(svms_stats_with_table, "avg")
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
                        lambda svm_name_and_stats_pair: f"Average coverage of {svm_name_and_stats_pair[0]} = {svm_name_and_stats_pair[1].avg}, {self._failed_maps_dict.get(svm_name_and_stats_pair[0], FailedMaps())}\n",
                        svms_stats,
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
