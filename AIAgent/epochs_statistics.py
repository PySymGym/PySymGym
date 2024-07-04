from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from statistics import mean
from typing import TypeAlias

import natsort
import pandas as pd
from common.typealias import SVMName
from common.game import GameMap
from common.classes import Map2Result

EpochNumber: TypeAlias = int


def sort_dict(d):
    return dict(natsort.natsorted(d.items()))


@dataclass
class TrainingParams:
    batch_size: int
    lr: float
    num_hops_1: int
    num_hops_2: int
    num_of_state_features: int
    epochs: int


@dataclass
class StatsWithTable:
    avg: float
    df: pd.DataFrame


class Status:

    def __init__(self):
        self.epoch: EpochNumber = 0
        self.failed_maps: list[GameMap] = []
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

        self._epochs_info: dict[EpochNumber, dict[SVMName, StatsWithTable]] = {}
        self._epoch_number: EpochNumber = 0
        self._svms_status: dict[SVMName, Status] = {}

    def update_file(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            self.__update_file()
            return res

        return wrapper

    @update_file
    def start_training_session(
        self,
        batch_size: int,
        lr: float,
        num_hops_1: int,
        num_hops_2: int,
        num_of_state_features: int,
        epochs: int,
    ):
        self._training_params: TrainingParams = TrainingParams(
            batch_size, lr, num_hops_1, num_hops_2, num_of_state_features, epochs
        )

    @update_file
    def fail(self, game_maps: list[GameMap]):
        for game_map in game_maps:
            svm_name = game_map.SVMInfo.name
            svm_status: Status = self._svms_status.get(svm_name, Status())
            svm_status.failed_maps.append(game_map)
            svm_status.count_of_failed_maps += 1
            self._svms_status[svm_name] = svm_status

    def __clear_failed_maps(self):
        for svm_status in self._svms_status.values():
            svm_status.failed_maps.clear()

    def get_failed_maps(self) -> list[GameMap]:
        """
        Returns failed maps on total epoch.

        NB: The list of failed maps is cleared after each request.
        """
        total_failed_maps: list[GameMap] = []
        for svm_status in self._svms_status.values():
            total_failed_maps.extend(svm_status.failed_maps)
        self.__clear_failed_maps()
        return total_failed_maps

    @update_file
    def update_results(
        self,
        average_result: float,
        map2results_list: list[Map2Result],
    ):
        epoch_info = self._epochs_info.get(self._epoch_number, {})
        svms_and_map2results_lists: dict[SVMName, list[Map2Result]] = dict()
        for map2result in map2results_list:
            svm_name = map2result.map.SVMInfo.name
            map2results_list_of_svm = svms_and_map2results_lists.get(svm_name, [])
            map2results_list_of_svm.append(map2result)
            svms_and_map2results_lists[svm_name] = map2results_list_of_svm
        for svm_name, map2results_list in svms_and_map2results_lists.items():
            epoch_info[svm_name] = StatsWithTable(
                average_result, convert_to_df(map2results_list)
            )
        self._epochs_info[self._epoch_number] = sort_dict(epoch_info)

        self._epoch_number += 1
        for svm_name in svms_and_map2results_lists:
            svm_status = self._svms_status.get(svm_name, Status())
            svm_status.epoch = self._epoch_number
            self._svms_status[svm_name] = svm_status

    def __get_training_info(self) -> str:
        def get_svms_info():
            info = ""
            for svm_name, status in self._svms_status.items():
                info += f"{svm_name}: {str(status)}\n"
            return info

        def get_training_params_info():
            training_params = self._training_params
            training_params_info = (
                f"epochs={training_params.epochs}, "
                f"batch_size={training_params.batch_size}, "
                f"lr={training_params.lr}, "
                f"num_hops_1={training_params.num_hops_1}, "
                f"num_hops_2={training_params.num_hops_2}, "
                f"num_of_state_features={training_params.num_of_state_features}"
            )
            return training_params_info

        return f"{get_training_params_info()}\n{get_svms_info()}"

    def __get_epochs_results(self) -> str:
        epochs_results = str()
        for epoch, v in self._epochs_info.items():
            avgs = list(map(lambda statsWithTable: statsWithTable.avg, v.values()))
            avg_common = mean(avgs)
            epoch_results = list(
                map(lambda statsWithTable: statsWithTable.df, v.values())
            )
            df = pd.concat(epoch_results, axis=1)
            epochs_results += f"Epoch#{epoch} Average coverage: {str(avg_common)}\n"
            names_and_averages = zip(v.keys(), avgs)
            epochs_results += "".join(
                list(
                    map(
                        lambda pair: f"Average coverage of {pair[0]} = {pair[1]}\n",
                        names_and_averages,
                    )
                )
            )
            epochs_results += df.to_markdown(tablefmt="psql") + "\n"
        return epochs_results

    def __update_file(self):
        svms_info = self.__get_training_info()
        epochs_results = self.__get_epochs_results()
        with open(self._file, "w") as f:
            f.write(svms_info)
            f.write(epochs_results)


def convert_to_df(map2result_list: list[Map2Result]) -> pd.DataFrame:
    maps = []
    results = []
    for map2result in map2result_list:
        _map = map2result.map
        map_name = _map.MapName
        game_result_str = map2result.game_result.printable(verbose=True)
        maps.append(f"{_map.SVMInfo.name} : {map_name}")
        results.append(game_result_str)

    df = pd.DataFrame(results, columns=["Game result"], index=maps).T

    return df
