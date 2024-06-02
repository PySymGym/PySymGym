from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Optional, TypeAlias

import natsort
import pandas as pd
from common.classes import Map2Result

EpochNumber: TypeAlias = int
SVMName: TypeAlias = str


def sort_dict(d):
    return dict(natsort.natsorted(d.items()))


@dataclass
class TrainingParams:
    batch_size: int
    lr: float
    num_hops_1: int
    num_hops_2: int
    num_of_state_features: int


@dataclass
class StatsWithTable:
    avg: float
    df: pd.DataFrame


class SVMStatus(Enum):
    RUNNING = "running"
    FAILED = "failed"
    FINISHED = "finished"


@dataclass
class Status:
    """
    status : SVMStatus.RUNNING | SVMStatus.FAILED | SVMStatus.FINISHED
    """

    status: SVMStatus
    epoch: int

    def __str__(self) -> str:
        result: str = f"status={self.status.value}"
        if self.status == SVMStatus.FAILED:
            result += f", on epoch = {self.epoch}"
        return result


class StatisticsCollector:
    def __init__(
        self,
        svm_count: int,
        file: Path,
    ):
        self._SVM_count: int = svm_count
        self._file = file

        self._svms_info: dict[SVMName, Optional[TrainingParams]] = {}
        self._epochs: dict[SVMName, Optional[EpochNumber]] = {}
        self._sessions_info: dict[EpochNumber, dict[SVMName, StatsWithTable]] = {}
        self._status: dict[SVMName, Status] = {}

        self._running: SVMName | None = None

    def register_new_training_session(self, svm_name: SVMName):
        self._running = svm_name
        self._svms_info[svm_name] = None
        self._epochs[svm_name] = None
        self._svms_info = sort_dict(self._svms_info)
        self._update_file()

    def start_training_session(
        self,
        batch_size: int,
        lr: float,
        num_hops_1: int,
        num_hops_2: int,
        num_of_state_features: int,
        epochs: int,
    ):
        svm_name = self._running
        self._epochs[svm_name] = epochs
        self._status[svm_name] = Status(SVMStatus.RUNNING, 0)

        self._svms_info[svm_name] = TrainingParams(
            batch_size, lr, num_hops_1, num_hops_2, num_of_state_features
        )
        self._update_file()

    def fail(self):
        svm_name = self._running
        self._status[svm_name].status = SVMStatus.FAILED
        self._running = None
        self._update_file()

    def finish(self):
        svm_name = self._running
        self._status[svm_name].status = SVMStatus.FINISHED
        self._running = None
        self._update_file()

    def update_results(
        self,
        average_result: float,
        map2results_list: list[Map2Result],
    ):
        svm_name = self._running
        epoch = self._status[svm_name].epoch

        results = self._sessions_info.get(epoch, {})
        results[svm_name] = StatsWithTable(
            average_result, convert_to_df(svm_name, map2results_list)
        )
        self._sessions_info[epoch] = sort_dict(results)
        self._status[svm_name].epoch += 1
        self._update_file()

    def _get_training_info(self) -> str:
        def svm_info_line(svm_info):
            svm_name, training_params = svm_info[0], svm_info[1]
            epochs = self._epochs[svm_name]
            status: Optional[Status] = self._status.get(svm_name, None)
            if status is None:
                return ""

            svm_info_line = (
                f"{svm_name} : "
                f"{str(status)}, "
                f"epochs={epochs}, "
                f"batch_size={training_params.batch_size}, "
                f"lr={training_params.lr}, "
                f"num_hops_1={training_params.num_hops_1}, "
                f"num_hops_2={training_params.num_hops_2}, "
                f"num_of_state_features={training_params.num_of_state_features}\n"
            )
            return svm_info_line

        return "".join(list(map(svm_info_line, self._svms_info.items())))

    def _get_epochs_results(self) -> str:
        epochs_results = str()
        for epoch, v in self._sessions_info.items():
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

    def _update_file(self):
        svms_info = self._get_training_info()
        epochs_results = self._get_epochs_results()
        with open(self._file, "w") as f:
            f.write(svms_info)
            f.write(epochs_results)


def convert_to_df(svm_name: SVMName, map2result_list: list[Map2Result]) -> pd.DataFrame:
    maps = []
    results = []
    for map2result in map2result_list:
        map_name = map2result.map.MapName
        game_result_str = map2result.game_result.printable(verbose=True)
        maps.append(f"{svm_name} : {map_name}")
        results.append(game_result_str)

    df = pd.DataFrame(results, columns=["Game result"], index=maps).T

    return df
