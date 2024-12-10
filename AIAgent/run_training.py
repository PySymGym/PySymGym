import argparse
import csv
import json
import logging
import random
import numpy as np
from ml.inference import TORCH
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Optional
from torch_geometric.data import Dataset

import joblib
import mlflow
import optuna
import torch
import yaml
from common.classes import GameFailed
from ml.training.statistics import get_svms_statistics, AVERAGE_COVERAGE
from common.config import (
    Config,
    OptunaConfig,
    TrainingConfig,
    ValidationConfig,
    ValidationWithLoss,
    ValidationWithSVMs,
)
from common.game import GameMap, GameMap2SVM
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
    CURRENT_TABLE_PATH,
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


def get_maps(validation_with_svms_config: ValidationWithSVMs):
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


def l2_norm(data):
        scaled_data = data.clone()
        attr = data[TORCH.gamevertex_history_statevertex].edge_attr.to(torch.float)
        scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = torch.nn.functional.normalize(
            attr, dim=0, p=2
        )
        return scaled_data

def l_inf_norm(data):
    scaled_data = data.clone()
    attr = data[TORCH.gamevertex_history_statevertex].edge_attr.to(torch.float)
    scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = attr / (
        torch.max(attr) + 1e-12
    )
    return scaled_data

def min_max_scaling(data):
    scaled_data = data.clone()
    attr = data[TORCH.gamevertex_history_statevertex].edge_attr.to(torch.float)
    scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = (
        attr - torch.min(attr)
    ) / (
        torch.max(attr)
        - torch.min(attr)
        + 1e-12
    )
    return scaled_data

def z_score_norm(data):
    scaled_data = data.clone()
    attr = data[TORCH.gamevertex_history_statevertex].edge_attr.to(torch.float)
    mean = torch.mean(attr, dim=0)
    std = torch.std(attr, dim=0) + 1e-12
    scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = (attr - mean) / std
    return scaled_data

def max_abs_scaling(data):
    scaled_data = data.clone()
    attr = data[TORCH.gamevertex_history_statevertex].edge_attr.to(torch.float)
    max_abs = torch.max(torch.abs(attr), dim=0).values + 1e-12
    scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = attr / max_abs
    return scaled_data

def log_scaling(data):
    scaled_data = data.clone()
    attr = data[TORCH.gamevertex_history_statevertex].edge_attr.to(torch.float)
    scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = torch.log1p(attr)
    return scaled_data

def robust_scaling(data):
    scaled_data = data.clone()
    edge_attr = data[TORCH.gamevertex_history_statevertex].edge_attr.to(torch.float)
    median = torch.median(edge_attr, dim=0).values
    q1 = torch.quantile(edge_attr, 0.25, dim=0)
    q3 = torch.quantile(edge_attr, 0.75, dim=0)
    iqr = q3 - q1 + 1e-12
    scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = (edge_attr - median) / iqr
    return scaled_data

def reciprocal_norm(data):
    scaled_data = data.clone()
    edge_attr = data[TORCH.gamevertex_history_statevertex].edge_attr.to(torch.float)
    
    epsilon = 1e-12
    edge_attr_normalized = 1 - 1 / (edge_attr + epsilon)
    
    scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = edge_attr_normalized
    return scaled_data


def run_training(
    optuna_config: OptunaConfig,
    training_config: TrainingConfig,
    validation_config: ValidationConfig,
    weights_uri: Optional[str],
):
    normalization_functions = [None,
                           l2_norm,
                           l_inf_norm,
                           min_max_scaling,
                           z_score_norm,
                           max_abs_scaling,
                           log_scaling,
                           reciprocal_norm]
    results = {}

    for normalization in normalization_functions:
        print(f"Running with transform function: {normalization.__name__ if normalization else 'None'}")

        def criterion_init():
            return nn.KLDivLoss(reduction="batchmean")

        if isinstance(validation_config.validation, ValidationWithLoss):

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

        elif isinstance(validation_config.validation, ValidationWithSVMs):
            maps: list[GameMap2SVM] = get_maps(validation_config.validation)
            with open(CURRENT_TABLE_PATH, "w") as statistics_file:
                statistics_writer = csv.DictWriter(
                    statistics_file,
                    sorted([game_map2svm.GameMap.MapName for game_map2svm in maps]),
                )
                statistics_writer.writeheader()

            def validate(model, dataset: TrainingDataset):
                map2results = validate_coverage(
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
            raw_dir=RAW_DATASET_PATH,
            processed_dir=PROCESSED_DATASET_PATH,
            train_percentage=training_config.train_percentage,
            threshold_steps_number=training_config.threshold_steps_number,
            load_to_cpu=training_config.load_to_cpu,
            threshold_coverage=training_config.threshold_coverage,
            transform_func=normalization,
        )

        def model_init(**model_params) -> nn.Module:
            state_model_encoder = StateModelEncoder(**model_params)
            if weights_uri is None:
                return state_model_encoder
            else:
                downloaded_artifact_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=weights_uri, dst_path=REPORT_PATH
                )
                state_model_encoder.load_state_dict(torch.load(downloaded_artifact_path, map_location=GeneralConfig.DEVICE))
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
        if optuna_config.study_uri is None and weights_uri is None:

            def save_study(study, _):
                joblib.dump(study, CURRENT_STUDY_PATH)
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
            best_value = study.best_value
            results[normalization.__name__ if normalization else "None"] = best_value
        else:
            downloaded_artifact_path = mlflow.artifacts.download_artifacts(
                optuna_config.study_uri, dst_path=str(REPORT_PATH)
            )
            study: optuna.Study = joblib.load(downloaded_artifact_path)
            for _ in range(optuna_config.n_trials):
                objective_partial(study.best_trial)
            best_value = study.best_value
            results[normalization.__name__ if normalization else "None"] = best_value

    print("Results for all transform functions:")
    for name, result in results.items():
        print(f"{name}: {result}")
    best_normalization = min(results, key=results.get)
    print(f"Best normalization function: {best_normalization} with result {results[best_normalization]}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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

    g = torch.Generator()
    g.manual_seed(42) 

    config = TrialSettings(
        lr=0.0006347818494377509,
        batch_size=628,
        num_hops_1=6,
        num_hops_2=10,
        num_of_state_features=15,
        hidden_channels=120,
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

    with mlflow.start_run():
        mlflow.log_params(asdict(config))
        for epoch in range(epochs):
            dataset.switch_to("train")
            train_dataloader = DataLoader(
                dataset, 
                config.batch_size, 
                shuffle=True, 
                worker_init_fn=seed_worker,
                generator=g,
                )
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

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

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
