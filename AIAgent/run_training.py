import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Optional

import joblib
import mlflow
import optuna
import torch
import yaml
from torch import nn
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from common.classes import GameFailed
from common.config import (
    Config,
    OptunaConfig,
    TrainingConfig,
    ValidationConfig,
    ValidationLoss,
    ValidationSVMViaServer,
)
from common.game import GameMap, GameMap2SVM
from config import GeneralConfig
from ml.dataset import TrainingDataset
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2Parametrized.model import (
    StateModelEncoder,
)
from ml.training.early_stopping import EarlyStopping
from ml.training.train import train
from common.file_system_utils import create_folders_if_necessary, create_file
from ml.validation.statistics import get_svms_statistics, AVERAGE_COVERAGE
from ml.validation.validate_coverage_via_server import (
    validate_coverage_via_server,
)
from ml.validation.validate_loss import validate_loss
from paths import (
    LOG_PATH,
    PROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
    CURRENT_MODEL_PATH,
    CURRENT_STUDY_PATH,
    REPORT_PATH,
    CURRENT_TABLE_PATH,
    CURRENT_TRIAL_PATH,
)

logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)


create_folders_if_necessary([PROCESSED_DATASET_PATH])


def get_maps(validation_with_svms_config: ValidationSVMViaServer):
    maps: list[GameMap2SVM] = list()
    for platform in validation_with_svms_config.platforms_config:
        for svm_info in platform.svms_info:
            for dataset_config in platform.dataset_configs:
                dataset_base_path = dataset_config.dataset_base_path
                dataset_description = dataset_config.dataset_description
                with open(dataset_description, "r") as maps_json:
                    single_json_maps: list[GameMap] = GameMap.schema().load(
                        json.loads(maps_json.read()), many=True
                    )
                single_json_maps_with_svms: list[GameMap2SVM] = list()
                for _map in single_json_maps:
                    fullName = os.path.join(dataset_base_path, _map.AssemblyFullName)
                    _map.AssemblyFullName = fullName
                    single_json_maps_with_svms.append(GameMap2SVM(_map, svm_info))
                maps.extend(single_json_maps_with_svms)
    return maps


@dataclass
class TrialSettings:
    lr: float
    batch_size: int
    num_hops_1: int
    num_hops_2: int
    num_of_state_features: int
    hidden_channels: int
    normalization: bool
    early_stopping_state_len: int
    tolerance: float


def run_training(
    optuna_config: OptunaConfig,
    training_config: TrainingConfig,
    validation_config: ValidationConfig,
    weights_uri: Optional[str],
):
    def criterion_init():
        return nn.KLDivLoss(reduction="batchmean")

    if isinstance(validation_config.validation, ValidationLoss):

        def validate(model, dataset):
            criterion = criterion_init()
            result = validate_loss(
                model,
                dataset,
                criterion,
                validation_config.validation.batch_size,
            )
            metric_name = str(criterion).replace("(", "_").replace(")", "_")
            metrics = {metric_name: result}
            return result, metrics

    elif isinstance(validation_config.validation, ValidationSVMViaServer):
        maps: list[GameMap2SVM] = get_maps(validation_config.validation)
        with open(CURRENT_TABLE_PATH, "w") as statistics_file:
            statistics_writer = csv.DictWriter(
                statistics_file,
                sorted([game_map2svm.GameMap.MapName for game_map2svm in maps]),
            )
            statistics_writer.writeheader()

        def validate(model, dataset: TrainingDataset):
            map2results = validate_coverage_via_server(
                model, dataset, maps, validation_config.validation
            )
            metrics = get_svms_statistics(
                map2results, validation_config.validation, dataset
            )
            mlflow.log_artifact(CURRENT_TABLE_PATH)

            for map2result in map2results:
                if (
                    isinstance(map2result.game_result, GameFailed)
                    and validation_config.validation.fail_immediately
                ):
                    raise RuntimeError("Validation failed")
            return metrics[AVERAGE_COVERAGE], metrics

    dataset = TrainingDataset(
        RAW_DATASET_PATH,
        PROCESSED_DATASET_PATH,
        train_percentage=training_config.train_percentage,
        threshold_steps_number=training_config.threshold_steps_number,
        load_to_cpu=training_config.load_to_cpu,
        threshold_coverage=training_config.threshold_coverage,
    )

    def model_init(**model_params) -> nn.Module:
        state_model_encoder = StateModelEncoder(**model_params)
        if weights_uri is None:
            return state_model_encoder
        else:
            downloaded_artifact_path = mlflow.artifacts.download_artifacts(
                artifact_uri=weights_uri, dst_path=REPORT_PATH
            )
            state_model_encoder.load_state_dict(torch.load(downloaded_artifact_path))
            return state_model_encoder

    objective_partial = partial(
        objective,
        dataset=dataset,
        dynamic_dataset=training_config.dynamic_dataset,
        model_init=model_init,
        criterion_init=criterion_init,
        epochs=training_config.epochs,
        validate=validate,
    )
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=optuna_config.n_startup_trials
    )
    if optuna_config.trial_uri is None and weights_uri is None:

        def save_study_and_trial(study, trial):
            joblib.dump(study, CURRENT_STUDY_PATH)
            joblib.dump(trial, CURRENT_TRIAL_PATH)
            with mlflow.start_run(mlflow.last_active_run().info.run_id):
                mlflow.log_artifact(CURRENT_STUDY_PATH)
                mlflow.log_artifact(CURRENT_TRIAL_PATH)
                mlflow.set_tag("best_trial_number", study.best_trial.number)

        study = optuna.create_study(
            sampler=sampler, direction=optuna_config.study_direction
        )
        study.optimize(
            objective_partial,
            n_trials=optuna_config.n_trials,
            gc_after_trial=True,
            n_jobs=optuna_config.n_jobs,
            callbacks=[save_study_and_trial],
        )
    else:
        downloaded_artifact_path = mlflow.artifacts.download_artifacts(
            optuna_config.trial_uri, dst_path=str(REPORT_PATH)
        )
        trial: optuna.trial.FrozenTrial = joblib.load(downloaded_artifact_path)
        for _ in range(optuna_config.n_trials):
            objective_partial(trial)


def objective(
    trial: optuna.Trial,
    dataset: TrainingDataset,
    dynamic_dataset: bool,
    model_init: Callable[[Any], nn.Module],
    criterion_init: Callable,
    epochs: int,
    validate: Callable[
        [nn.Module, Dataset], tuple[int | float, dict[str, int | float]]
    ],
):
    config = TrialSettings(
        lr=trial.suggest_float("lr", 1e-7, 1e-3),
        batch_size=trial.suggest_int("batch_size", 8, 800),
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
    model: nn.Module = model_init(
        hidden_channels=config.hidden_channels,
        num_of_state_features=config.num_of_state_features,
        num_hops_1=config.num_hops_1,
        num_hops_2=config.num_hops_2,
        normalization=config.normalization,
    )
    model.to(GeneralConfig.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = criterion_init()

    with mlflow.start_run(run_name=str(trial.number)):
        mlflow.log_params(asdict(config))
        for epoch in range(epochs):
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
            result, metrics = validate(model, dataset)
            mlflow.log_metrics(metrics, step=epoch)
            if dynamic_dataset:
                dataset.update_meta_data()
            if not early_stopping.is_continue(result):
                print(f"Training was stopped on {epoch} epoch.")
                break
            torch.cuda.empty_cache()
    return result


def main(config: str):
    with open(config, "r") as file:
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
