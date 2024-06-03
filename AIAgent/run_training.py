import argparse
import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional

import joblib
import numpy as np
import optuna
import torch
import yaml
from common.classes import Config, DatasetConfig, PlatformName, SVMConfig
from common.game import GameMap
from config import GeneralConfig, TrainingConfig
from epochs_statistics import StatisticsCollector
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2Parametrized.model import (
    StateModelEncoder,
)
from ml.training.dataset import TrainingDataset
from ml.training.training import train
from ml.training.utils import create_file, create_folders_if_necessary
from ml.training.validation import validate_coverage
from paths import (
    LOG_PATH,
    PROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
    TRAINED_MODELS_PATH,
    TRAINING_RESULTS_PATH,
)
from torch import nn
from torch_geometric.loader import DataLoader

logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)


create_folders_if_necessary([TRAINING_RESULTS_PATH, PROCESSED_DATASET_PATH])


@dataclass
class TrialSettings:
    lr: float
    epochs: int
    batch_size: int
    optimizer: torch.optim.Optimizer
    loss: any
    random_seed: int
    num_hops_1: int
    num_hops_2: int
    num_of_state_features: int
    hidden_channels: int
    normalization: bool


class EpochsInfo:
    total_epochs: int = 0

    def next(self):
        self.total_epochs += 1


def model_saver(epochs_info: EpochsInfo, svm_name: str, dir: Path, model: object):
    """
    Use it to save your torch model with some info in the following format: "{`total_epochs` + `epoch`}_{`svm_name`}" in `dir` directory

    Parameters
    ----------
    :param EpochsInfo `epochs_info`: EpochsInfo's instance
    :param str `svm_name`: name of svm
    :param object `model`: model to save
    :param Path `dir`: directory

    """
    path_to_model = os.path.join(dir, f"{epochs_info.total_epochs}_{svm_name}")
    torch.save(model, Path(path_to_model))


def run_training(
    svm_configs: list[SVMConfig],
    datasets_of_platforms: dict[PlatformName, DatasetConfig],
    n_startup_trials: int,
    n_trials: int,
    path_to_weights: Optional[Path],
    logfile_base_name: str,
    statistics_collector: StatisticsCollector,
):
    models_path = os.path.join(TRAINED_MODELS_PATH, logfile_base_name)
    os.makedirs(models_path)

    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    study = optuna.create_study(
        sampler=sampler, direction=TrainingConfig.STUDY_DIRECTION
    )
    epochs_info = EpochsInfo()
    for svm_config in svm_configs:
        svm_info = svm_config.SVMInfo
        dataset_config = datasets_of_platforms[svm_config.platform_name]
        dataset_base_path = dataset_config.dataset_base_path
        dataset_description = dataset_config.dataset_description

        with open(dataset_description, "r") as maps_json:
            maps: List[GameMap] = GameMap.schema().load(
                json.loads(maps_json.read()), many=True
            )
            for _map in maps:
                fullName = os.path.join(dataset_base_path, _map.AssemblyFullName)
                _map.AssemblyFullName = fullName
                _map.SVMInfo = svm_info.create_single_svm_info()

        dataset = TrainingDataset(
            RAW_DATASET_PATH,
            PROCESSED_DATASET_PATH,
            maps,
            train_percentage=TrainingConfig.TRAIN_PERCENTAGE,
            threshold_steps_number=TrainingConfig.THRESHOLD_STEPS_NUMBER,
            load_to_cpu=TrainingConfig.LOAD_TO_CPU,
            threshold_coverage=TrainingConfig.THRESHOLD_COVERAGE,
        )

        def load_weights(model: torch.nn.Module):
            model.load_state_dict(torch.load(path_to_weights))
            return model

        def model_init(**model_params):
            state_model_encoder = StateModelEncoder(**model_params)
            if path_to_weights is None:
                return state_model_encoder
            return load_weights(state_model_encoder)

        statistics_collector.register_new_training_session(svm_info.name)

        objective_partial = partial(
            objective,
            statistics_collector=statistics_collector,
            dataset=dataset,
            dynamic_dataset=TrainingConfig.DYNAMIC_DATASET,
            model_init=model_init,
            epochs=svm_config.epochs,
            epochs_info=epochs_info,
            model_saver=partial(
                model_saver,
                epochs_info=epochs_info,
                svm_name=svm_info.name,
                dir=models_path,
            ),
            server_count=svm_info.count,
        )
        try:
            study.optimize(
                objective_partial,
                n_trials=n_trials,
                gc_after_trial=True,
                n_jobs=TrainingConfig.OPTUNA_N_JOBS,
            )
        except RuntimeError:
            logging.error(f"Fail to train with {svm_info.name}")
            statistics_collector.fail()
    joblib.dump(
        study,
        f"{logfile_base_name}.pkl",
    )


def objective(
    trial: optuna.Trial,
    statistics_collector: StatisticsCollector,
    dataset: TrainingDataset,
    dynamic_dataset: bool,
    model_init: Callable,
    epochs: int,
    epochs_info: EpochsInfo,
    model_saver: Callable,
    server_count: int,
):
    config = TrialSettings(
        lr=0.0003,  # trial.suggest_float("lr", 1e-7, 1e-3),
        batch_size=109,  # trial.suggest_int("batch_size", 8, 1800),
        epochs=epochs,
        optimizer=trial.suggest_categorical("optimizer", [torch.optim.Adam]),
        loss=trial.suggest_categorical("loss", [nn.KLDivLoss]),
        random_seed=937,
        num_hops_1=5,  # trial.suggest_int("num_hops_1", 2, 10),
        num_hops_2=4,  # trial.suggest_int("num_hops_2", 2, 10),
        num_of_state_features=30,  # trial.suggest_int("num_of_state_features", 8, 64),
        hidden_channels=110,  # trial.suggest_int("hidden_channels", 64, 128),
        normalization=True,
    )

    model = model_init(
        hidden_channels=config.hidden_channels,
        num_of_state_features=config.num_of_state_features,
        num_hops_1=config.num_hops_1,
        num_hops_2=config.num_hops_2,
        normalization=config.normalization,
    )
    model.to(GeneralConfig.DEVICE)

    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    criterion = config.loss()
    statistics_collector.start_training_session(
        batch_size=config.batch_size,
        lr=config.lr,
        num_hops_1=config.num_hops_1,
        num_hops_2=config.num_hops_2,
        num_of_state_features=config.num_of_state_features,
        epochs=config.epochs,
    )

    np.random.seed(config.random_seed)
    for _ in range(config.epochs):
        dataset.switch_to("train")
        train_dataloader = DataLoader(dataset, config.batch_size, shuffle=True)
        model.train()
        train(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )
        torch.cuda.empty_cache()
        model_saver(model=model.state_dict())

        model.eval()
        dataset.switch_to("val")
        result = validate_coverage(
            statistics_collector=statistics_collector,
            model=model,
            dataset=dataset,
            server_count=server_count,
        )
        if dynamic_dataset:
            dataset.update_meta_data()
        epochs_info.next()
    statistics_collector.finish()
    return result


def main(config: str):
    with open(config, "r") as file:
        trainings_parameters = yaml.safe_load(file)
    config: Config = Config(**trainings_parameters)

    timestamp = datetime.now().timestamp()
    logfile_base_name = f"{datetime.fromtimestamp(timestamp)}_Adam_KLDL"
    results_table_path = os.path.join(TRAINING_RESULTS_PATH, logfile_base_name + ".log")
    create_file(LOG_PATH)

    mp.set_start_method("spawn", force=True)
    print(GeneralConfig.DEVICE)
    statistics_collector = StatisticsCollector(results_table_path)

    n_startup_trials = config.OptunaConfig.n_startup_trials
    n_trials = config.OptunaConfig.n_trials
    path_to_weights = config.path_to_weights
    if path_to_weights is not None:
        path_to_weights = Path(path_to_weights).absolute()

    datasets_of_platforms: dict[PlatformName, DatasetConfig] = {}
    for platform in config.Platforms:
        datasets_of_platforms[platform.name] = platform.DatasetConfig

    run_training(
        svm_configs=config.SVMConfigs,
        datasets_of_platforms=datasets_of_platforms,
        n_startup_trials=n_startup_trials,
        n_trials=n_trials,
        path_to_weights=path_to_weights,
        logfile_base_name=logfile_base_name,
        statistics_collector=statistics_collector,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using configuration from a .yml file."
    )

    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )
    args = parser.parse_args()
    config = args.config
    main(config)
