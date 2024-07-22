import argparse
import json
import logging
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Optional

import joblib
import mlflow
import numpy as np
import optuna
import torch
import yaml
from common.config import (
    Config,
    OptunaConfig,
    Platform,
    TrainingConfig,
    ValidationConfig,
    ValidationWithLoss,
    ValidationWithSVMs,
)
from common.game import GameMap
from config import GeneralConfig
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2Parametrized.model import (
    StateModelEncoder,
)
from ml.training.dataset import TrainingDataset
from ml.training.early_stopping import EarlyStopping
from ml.training.train import train
from ml.training.utils import create_file, create_folders_if_necessary
from ml.training.validation import validate_coverage, validate_loss
from paths import (
    LOG_PATH,
    PROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
    CURRENT_MODEL_PATH,
    CURRENT_STUDY_PATH,
    REPORT_PATH,
)
from torch import nn
from torch_geometric.loader import DataLoader

logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)


create_folders_if_necessary([PROCESSED_DATASET_PATH])


@dataclass
class TrialSettings:
    lr: float
    epochs: int
    batch_size: int
    random_seed: int
    num_hops_1: int
    num_hops_2: int
    num_of_state_features: int
    hidden_channels: int
    normalization: bool
    early_stopping_state_len: int
    tolerance: float


def run_training(
    platforms_config: list[Platform],
    optuna_config: OptunaConfig,
    training_config: TrainingConfig,
    validation_config: ValidationConfig,
    weights_uri: Optional[str],
):
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=optuna_config.n_startup_trials
    )

    maps: list[GameMap] = list()
    for platform in platforms_config:
        svms_info = platform.svms_info
        for svm_info in svms_info:
            for dataset_config in platform.dataset_configs:
                dataset_base_path = dataset_config.dataset_base_path
                dataset_description = dataset_config.dataset_description
                with open(dataset_description, "r") as maps_json:
                    single_json_maps: list[GameMap] = GameMap.schema().load(
                        json.loads(maps_json.read()), many=True
                    )
                for _map in single_json_maps:
                    fullName = os.path.join(dataset_base_path, _map.AssemblyFullName)
                    _map.AssemblyFullName = fullName
                    _map.SVMInfo = svm_info
                maps.extend(single_json_maps)

    dataset = TrainingDataset(
        RAW_DATASET_PATH,
        PROCESSED_DATASET_PATH,
        maps,
        train_percentage=training_config.train_percentage,
        threshold_steps_number=training_config.threshold_steps_number,
        load_to_cpu=training_config.load_to_cpu,
        threshold_coverage=training_config.threshold_coverage,
    )

    def load_weights(model: nn.Module):
        downloaded_artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri=weights_uri, dst_path=REPORT_PATH
        )
        model.load_state_dict(torch.load(downloaded_artifact_path))
        return model

    def model_init(**model_params) -> nn.Module:
        state_model_encoder = StateModelEncoder(**model_params)
        if weights_uri is None:
            return state_model_encoder
        return load_weights(state_model_encoder)

    objective_partial = partial(
        objective,
        dataset=dataset,
        dynamic_dataset=training_config.dynamic_dataset,
        model_init=model_init,
        epochs=training_config.epochs,
        val_config=validation_config,
    )
    try:
        if optuna_config.study_uri is None and weights_uri is None:

            def save_study(study, _):
                joblib.dump(
                    study,
                    CURRENT_STUDY_PATH,
                )
                with mlflow.start_run(mlflow.last_active_run().info.run_id):
                    mlflow.log_artifact(CURRENT_STUDY_PATH)

            study = optuna.create_study(
                sampler=sampler, direction=optuna_config.study_direction
            )
            study.optimize(
                objective_partial,
                n_trials=optuna_config.n_trials,
                gc_after_trial=True,
                n_jobs=optuna_config.n_jobs,
                callbacks=[save_study],
            )
        else:
            downloaded_artifact_path = mlflow.artifacts.download_artifacts(
                optuna_config.study_uri, dst_path=str(REPORT_PATH)
            )
            study: optuna.Study = joblib.load(downloaded_artifact_path)
            objective_partial(study.best_trial)
    except RuntimeError:
        logging.error("Fail to train")


def objective(
    trial: optuna.Trial,
    dataset: TrainingDataset,
    dynamic_dataset: bool,
    model_init: Callable[[Any], nn.Module],
    epochs: int,
    val_config: ValidationConfig,
):
    config = TrialSettings(
        lr=trial.suggest_float("lr", 1e-7, 1e-3),
        batch_size=trial.suggest_int("batch_size", 8, 800),
        epochs=epochs,
        random_seed=937,
        num_hops_1=trial.suggest_int("num_hops_1", 2, 10),
        num_hops_2=trial.suggest_int("num_hops_2", 2, 10),
        num_of_state_features=trial.suggest_int("num_of_state_features", 8, 64),
        hidden_channels=trial.suggest_int("hidden_channels", 64, 128),
        normalization=True,
        early_stopping_state_len=5,
        tolerance=0.0001,
    )
    early_stopping = EarlyStopping(
        state_len=config.early_stopping_state_len, tolerance=config.tolerance
    )
    model = model_init(
        hidden_channels=config.hidden_channels,
        num_of_state_features=config.num_of_state_features,
        num_hops_1=config.num_hops_1,
        num_hops_2=config.num_hops_2,
        normalization=config.normalization,
    )
    model.to(GeneralConfig.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.KLDivLoss(reduction="batchmean")

    if isinstance(val_config.validation, ValidationWithLoss):

        def validate(model, dataset, epoch):
            return validate_loss(
                model, dataset, epoch, criterion, val_config.validation.batch_size
            )
    elif isinstance(val_config.validation, ValidationWithSVMs):

        def validate(model, dataset: TrainingDataset, epoch):
            result, failed_maps = validate_coverage(
                model, dataset, epoch, val_config.validation.servers_count
            )
            for _map in failed_maps:
                dataset.maps.remove(_map)
            return result

    np.random.seed(config.random_seed)
    with mlflow.start_run():
        mlflow.log_params(asdict(config))
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
            torch.save(model.state_dict(), CURRENT_MODEL_PATH)
            mlflow.log_artifact(CURRENT_MODEL_PATH, str(epoch))

            model.eval()
            dataset.switch_to("val")
            result = validate(model, dataset, epoch)
            if dynamic_dataset:
                dataset.update_meta_data()
            if not early_stopping.is_continue(result):
                print(f"Training was stopped on {epoch} epoch.")
                break
            torch.cuda.empty_cache()
    return result


def main(config_path: str):
    with open(config_path, "r") as file:
        config: Config = Config(**yaml.safe_load(file))
    create_file(LOG_PATH)

    mp.set_start_method("spawn", force=True)
    print(GeneralConfig.DEVICE)

    mlflow_config = config.mlflow_config
    if mlflow_config.tracking_uri is not None:
        mlflow.set_tracking_uri(uri=mlflow_config.tracking_uri)
    mlflow.set_experiment(mlflow_config.experiment_name)
    mlflow.set_experiment_tags(asdict(config))
    weights_uri = config.weights_uri

    run_training(
        platforms_config=config.platforms_config,
        optuna_config=config.optuna_config,
        training_config=config.training_config,
        validation_config=config.validation_config,
        weights_uri=weights_uri,
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
