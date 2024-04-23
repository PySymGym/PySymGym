import argparse
import json
import logging
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, List
import joblib
import optuna
import torch
from torch import nn
import multiprocessing as mp
from common.game import GameMap
from config import GeneralConfig
from ml.training.dataset import TrainingDataset
from ml.training.paths import (
    PROCESSED_DATASET_PATH,
    TRAINING_RESULTS_PATH,
    RAW_DATASET_PATH,
    LOG_PATH,
    TRAINED_MODELS_PATH,
)
import numpy as np
from ml.training.utils import create_folders_if_necessary
from ml.training.training import train
from ml.training.validation import validate_coverage
from dataclasses import dataclass
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2Parametrized.model import (
    StateModelEncoder,
)
from torch_geometric.loader import DataLoader
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


def objective(
    trial: optuna.Trial,
    dataset: TrainingDataset,
    dynamic_dataset: bool,
    model_init: Callable,
):
    config = TrialSettings(
        lr=0.0003994891827606452,  # trial.suggest_float("lr", 1e-7, 1e-3),
        batch_size=109,  # trial.suggest_int("batch_size", 8, 1800),
        epochs=10,
        optimizer=trial.suggest_categorical("optimizer", [torch.optim.Adam]),
        loss=trial.suggest_categorical("loss", [nn.KLDivLoss]),
        random_seed=937,
        num_hops_1=5,  # trial.suggest_int("num_hops_1", 2, 10),
        num_hops_2=4,  # trial.suggest_int("num_hops_2", 2, 10),
        num_of_state_features=30,  # trial.suggest_int("num_of_state_features", 8, 64),
        hidden_channels=110,  # trial.suggest_int("hidden_channels", 64, 128),
    )

    model = model_init(
        **{
            "hidden_channels": config.hidden_channels,
            "num_of_state_features": config.num_of_state_features,
            "num_hops_1": config.num_hops_1,
            "num_hops_2": config.num_hops_2,
        }
    )
    model.to(GeneralConfig.DEVICE)

    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    criterion = config.loss()

    timestamp = datetime.now().timestamp()
    run_name = (
        f"{datetime.fromtimestamp(timestamp)}_{config.batch_size}"
        f"_Adam_{config.lr}_KLDL_{config.num_hops_1}_{config.num_hops_2}"
        f"_{config.num_of_state_features}_{config.epochs}"
    )
    print(run_name)
    path_to_trained_models = os.path.join(TRAINED_MODELS_PATH, run_name)
    os.makedirs(path_to_trained_models)
    create_file(LOG_PATH)
    results_table_path = Path(os.path.join(TRAINING_RESULTS_PATH, run_name + ".log"))
    create_file(results_table_path)

    np.random.seed(config.random_seed)
    for epoch in range(config.epochs):
        dataset.switch_to("train")
        train_dataloader = DataLoader(dataset, config.batch_size, shuffle=True)
        model.train()
        train(train_dataloader, model, optimizer, criterion)
        torch.cuda.empty_cache()
        path_to_model = os.path.join(TRAINED_MODELS_PATH, run_name, str(epoch + 1))
        torch.save(model.state_dict(), Path(path_to_model))

        model.eval()
        dataset.switch_to("val")
        result = validate_coverage(model, epoch, dataset, run_name)
        if dynamic_dataset:
            dataset.update_meta_data()
    return result


def main(args):
    DYNAMIC_DATASET = True
    TRAIN_PERCENTAGE = 1
    THRESHOLD_COVERAGE = 100
    THRESHOLD_STEPS_NUMBER = None
    LOAD_TO_CPU = False

    OPTUNA_N_JOBS = 1
    N_STARTUP_TRIALS = 10
    STUDY_DIRECTION = "maximize"
    N_TRIALS = 100

    dataset_base_path = str(Path(args.datasetbasepath).resolve())
    dataset_description = str(Path(args.datasetdescription).resolve())
    weighs_path = Path(args.weights_path).absolute() if args.weights_path else None

    print(GeneralConfig.DEVICE)

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
        train_percentage=TRAIN_PERCENTAGE,
        threshold_steps_number=THRESHOLD_STEPS_NUMBER,
        load_to_cpu=LOAD_TO_CPU,
        threshold_coverage=THRESHOLD_COVERAGE,
    )

    def load_weights(model: torch.nn.Module):
        model.load_state_dict(torch.load(weighs_path))
        return model

    if weighs_path is None:
        model_init = lambda **model_params: StateModelEncoder(**model_params)
    else:
        model_init = lambda **model_params: load_weights(
            StateModelEncoder(**model_params)
        )
    mp.set_start_method("spawn", force=True)

    sampler = optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(sampler=sampler, direction=STUDY_DIRECTION)

    objective_partial = partial(
        objective,
        dataset=dataset,
        dynamic_dataset=DYNAMIC_DATASET,
        model_init=model_init,
    )

    study.optimize(
        objective_partial, n_trials=N_TRIALS, gc_after_trial=True, n_jobs=OPTUNA_N_JOBS
    )
    joblib.dump(study, f"{datetime.fromtimestamp(datetime.now().timestamp())}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasetdescription",
        type=str,
        help="full paths to JSON-file with dataset description",
        required=True,
    )
    parser.add_argument(
        "--datasetbasepath",
        type=str,
        help="path to dir with explored dlls",
        required=True,
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="path to model weights to load",
        required=False,
    )
    args = parser.parse_args()
    main(args)
