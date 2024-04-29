import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, List
import joblib
import optuna
import numpy as np
from dataclasses import dataclass
from torch import nn
import torch
from torch_geometric.loader import DataLoader
from multiprocessing.managers import AutoProxy
import multiprocessing as mp
import yaml

from common.classes import SVMInfo
from common.game import GameMap
from config import GeneralConfig, TrainingConfig
from ml.training.dataset import TrainingDataset
from ml.training.paths import (
    PROCESSED_DATASET_PATH,
    TRAINING_RESULTS_PATH,
    RAW_DATASET_PATH,
    LOG_PATH,
    TRAINED_MODELS_PATH,
)
from epochs_statistics.classes import StatisticsCollector, StatisticsManager
from ml.training.utils import create_folders_if_necessary
from ml.training.training import train
from ml.training.validation import validate_coverage
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2Parametrized.model import (
    StateModelEncoder,
)
from ml.training.utils import create_file


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


def run_training(
    svm_info: SVMInfo,
    statistics_collector: "AutoProxy[StatisticsCollector]",
    dataset_base_path: Path,
    dataset_description: Path,
    n_startup_trials: int,
    n_trials: int,
    num_epochs: int,
    path_to_weights: str,
    run_name: str,
):
    with open(dataset_description, "r") as maps_json:
        maps: List[GameMap] = GameMap.schema().load(
            json.loads(maps_json.read()), many=True
        )
        for _map in maps:
            fullName = os.path.join(dataset_base_path, _map.AssemblyFullName)
            _map.AssemblyFullName = fullName

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

    if path_to_weights is None:
        model_init = lambda **model_params: StateModelEncoder(**model_params)
    else:
        model_init = lambda **model_params: load_weights(
            StateModelEncoder(**model_params)
        )

    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    study = optuna.create_study(
        sampler=sampler, direction=TrainingConfig.STUDY_DIRECTION
    )

    objective_partial = partial(
        objective,
        svm_info=svm_info,
        statistics_collector=statistics_collector,
        dataset=dataset,
        dynamic_dataset=TrainingConfig.DYNAMIC_DATASET,
        model_init=model_init,
        epochs=num_epochs,
        run_name=run_name,
    )

    study.optimize(
        objective_partial,
        n_trials=n_trials,
        gc_after_trial=True,
        n_jobs=TrainingConfig.OPTUNA_N_JOBS,
    )
    joblib.dump(study, f"{datetime.fromtimestamp(datetime.now().timestamp())}.pkl")


def objective(
    trial: optuna.Trial,
    svm_info: SVMInfo,
    statistics_collector: "AutoProxy[StatisticsCollector]",
    dataset: TrainingDataset,
    dynamic_dataset: bool,
    model_init: Callable,
    epochs: int,
    run_name: str,
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
        **{
            "hidden_channels": config.hidden_channels,
            "num_of_state_features": config.num_of_state_features,
            "num_hops_1": config.num_hops_1,
            "num_hops_2": config.num_hops_2,
            "normalization": config.normalization,
        }
    )
    model.to(GeneralConfig.DEVICE)

    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    criterion = config.loss()
    statistics_collector.register_training_session(
        svm_info.name,
        config.batch_size,
        config.lr,
        config.num_hops_1,
        config.num_hops_2,
        config.num_of_state_features,
        config.epochs,
    )

    run_name = (
        f"{run_name}_{svm_info.name}_"
        f"_{config.batch_size}_{config.lr}_{config.num_hops_1}_{config.num_hops_2}"
        f"_{config.num_of_state_features}_{config.epochs}"
    )
    models = os.path.join(TRAINED_MODELS_PATH, run_name)
    os.makedirs(models)

    np.random.seed(config.random_seed)
    for epoch in range(config.epochs):
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
        path_to_model = os.path.join(models, str(epoch + 1))
        torch.save(model.state_dict(), Path(path_to_model))

        model.eval()
        dataset.switch_to("val")
        result = validate_coverage(
            svm_info=svm_info,
            statistics_collector=statistics_collector,
            model=model,
            epoch=epoch,
            dataset=dataset,
        )
        if dynamic_dataset:
            dataset.update_meta_data()
    return result


def main(args):
    args = parser.parse_args()
    config = args.config

    with open(config, "r") as file:
        trainings_parameters = yaml.safe_load(file)

    training_count = len(trainings_parameters)

    timestamp = datetime.now().timestamp()
    run_name = f"{datetime.fromtimestamp(timestamp)}_Adam_KLDL"
    results_table_path = os.path.join(TRAINING_RESULTS_PATH, run_name + ".log")
    create_file(LOG_PATH)

    mp.set_start_method("spawn", force=True)
    print(GeneralConfig.DEVICE)
    with StatisticsManager() as manager:
        # shared object
        statistics_collector = manager.StatisticsCollector(
            training_count, results_table_path
        )
        with ProcessPoolExecutor(
            max_workers=training_count,
        ) as executor:
            for training_parameters in trainings_parameters:
                svm_info = SVMInfo.from_dict(training_parameters["SVMConfig"])
                dataset_base_path = str(
                    Path(
                        training_parameters["DatasetConfig"][
                            "dataset_base_path"
                        ]  # path to dir with explored dlls
                    ).resolve()
                )
                dataset_description = str(
                    Path(
                        training_parameters["DatasetConfig"][
                            "dataset_description"
                        ]  # full paths to JSON-file with dataset description
                    ).resolve()
                )
                n_startup_trials = int(
                    training_parameters["OptunaConfig"][
                        "n_startup_trials"
                    ]  # number of initial trials with random sampling for optuna's TPESampler
                )
                n_trials = int(
                    training_parameters["OptunaConfig"]["n_trials"]
                )  # number of optuna's trials
                num_epochs = int(
                    training_parameters["TrainConfig"]["epochs"]
                )  # number of epochs
                path_to_weights = training_parameters["TrainConfig"].get(
                    "path_to_weights", None
                )  # path to model weights to load
                if not path_to_weights is None:
                    path_to_weights = Path(path_to_weights).absolute()
                executor.submit(
                    run_training,
                    svm_info=svm_info,
                    statistics_collector=statistics_collector,
                    dataset_base_path=dataset_base_path,
                    dataset_description=dataset_description,
                    n_startup_trials=n_startup_trials,
                    n_trials=n_trials,
                    num_epochs=num_epochs,
                    path_to_weights=path_to_weights,
                    run_name=run_name,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using configuration from a .yml file."
    )

    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )
    args = parser.parse_args()
    main(args)
