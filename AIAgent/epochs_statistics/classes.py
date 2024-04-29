from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from pathlib import Path
import pandas as pd
from statistics import mean
import natsort
from epochs_statistics.tables import table_to_string
from epochs_statistics.common import EpochNumber, SVMName


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


class StatisticsCollector:
    _lock = mp.Lock()

    def __init__(
        self,
        SVM_count: int,
        file: Path,
    ):
        self._SVM_count: int = SVM_count
        self._file = file

        self._SVMS_info: dict[SVMName, TrainingParams] = {}
        self._sessions_info: dict[EpochNumber, dict[SVMName, StatsWithTable]] = {}

    def register_training_session(
        self,
        SVM_name: SVMName,
        batch_size: int,
        lr: float,
        num_hops_1: int,
        num_hops_2: int,
        num_of_state_features: int,
        epochs: int,
    ):
        self._SVMS_info[SVM_name] = TrainingParams(
            batch_size, lr, num_hops_1, num_hops_2, num_of_state_features, epochs
        )
        self._SVMS_info = sort_dict(self._SVMS_info)
        self._update_file()

    def update_results(
        self,
        epoch: EpochNumber,
        SVM_name: SVMName,
        average_result: float,
        df: pd.DataFrame,
    ):
        results = self._sessions_info.get(epoch, {})
        results[SVM_name] = StatsWithTable(average_result, df)
        self._sessions_info[epoch] = sort_dict(results)
        self._update_file()

    def _get_SVMS_info(self) -> str:
        svm_info_line = lambda svm_info: (
            f"{svm_info[0]} : "
            f"batch_size={svm_info[1].batch_size}, "
            f"lr={svm_info[1].lr}, "
            f"num_hops_1={svm_info[1].num_hops_1}, "
            f"num_hops_2={svm_info[1].num_hops_2}, "
            f"num_of_state_features={svm_info[1].num_of_state_features}, "
            f"epochs={svm_info[1].epochs}\n"
        )

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
            epochs_results += table_to_string(df) + "\n"
        return epochs_results

    def _update_file(self):
        with self._lock:
            SVMS_info = self._get_SVMS_info()
            epochs_results = self._get_epochs_results()
        with open(self._file, "w") as f:
            f.write(SVMS_info)
            f.write(epochs_results)


class StatisticsManager(BaseManager):
    pass


StatisticsManager.register("StatisticsCollector", StatisticsCollector)
