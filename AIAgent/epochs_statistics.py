from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import TypeAlias

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
    epochs: int


@dataclass
class StatsWithTable:
    avg: float
    df: pd.DataFrame


@dataclass
class Status:
    """
    status : RUNNING_STATUS | FAILED_STATUS | FINISHED_STATUS
    """

    status: str
    epoch: EpochNumber

    RUNNING_STATUS = "running"
    FAILED_STATUS = "failed"
    FINISHED_STATUS = "finished"

    def __str__(self) -> str:
        status = self.status
        result: str = f"status={self.status}"
        if status == self.FAILED_STATUS:
            result += f", on epoch = {self.epoch}"
        return result


class StatisticsCollector:
    def __init__(
        self,
        SVM_count: int,
        file: Path,
    ):
        self._SVM_count: int = SVM_count
        self._file = file

        self._SVMS_info: dict[SVMName, TrainingParams | None] = {}
        self._sessions_info: dict[EpochNumber, dict[SVMName, StatsWithTable]] = {}
        self._status: dict[SVMName, Status] = {}

        self._running: SVMName | None = None

    def register_new_training_session(self, SVM_name: SVMName):
        self._running = SVM_name
        self._SVMS_info[SVM_name] = None
        self._SVMS_info = sort_dict(self._SVMS_info)
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
        SVM_name = self._running
        self._status[SVM_name] = Status(Status.RUNNING_STATUS, 0)

        self._SVMS_info[SVM_name] = TrainingParams(
            batch_size,
            lr,
            num_hops_1,
            num_hops_2,
            num_of_state_features,
            epochs,
        )
        self._update_file()

    def fail(self):
        SVM_name = self._running
        self._status[SVM_name].status = Status.FAILED_STATUS
        self._running = None
        self._update_file()

    def finish(self):
        SVM_name = self._running
        self._status[SVM_name].status = Status.FINISHED_STATUS
        self._running = None
        self._update_file()

    def update_results(
        self,
        average_result: float,
        map2results_list: list[Map2Result],
    ):
        SVM_name = self._running
        epoch = self._status[SVM_name].epoch

        results = self._sessions_info.get(epoch, {})
        results[SVM_name] = StatsWithTable(
            average_result, convert_to_df(SVM_name, map2results_list)
        )
        self._sessions_info[epoch] = sort_dict(results)
        self._status[SVM_name].epoch = epoch + 1
        self._update_file()

    def _get_training_info(self) -> str:
        def svm_info_line(svm_info):
            svm_name, training_params = svm_info[0], svm_info[1]
            status: Status | None = self._status.get(svm_name, None)
            if status:
                svm_info_line = (
                    f"{svm_name} : "
                    f"{str(status)}, "
                    f"batch_size={training_params.batch_size}, "
                    f"lr={training_params.lr}, "
                    f"num_hops_1={training_params.num_hops_1}, "
                    f"num_hops_2={training_params.num_hops_2}, "
                    f"num_of_state_features={training_params.num_of_state_features}, "
                    f"epochs={training_params.epochs}\n"
                )
            else:
                svm_info_line = ""
            return svm_info_line

        return "".join(list(map(svm_info_line, self._SVMS_info.items())))

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
        SVMS_info = self._get_training_info()
        epochs_results = self._get_epochs_results()
        with open(self._file, "w") as f:
            f.write(SVMS_info)
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
