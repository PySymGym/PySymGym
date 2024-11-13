from pathlib import Path

import joblib
import optuna
from ml.inference import TORCH
from ml.training.validation import validate_loss
from config import GeneralConfig
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2Parametrized.model import (
    StateModelEncoder,
)
import mlflow
from ml.training.dataset import TrainingDataset
import torch
from torch import nn
from run_training import objective
from paths import REPORT_PATH
import numpy as np
import random

RAW_DATASET = Path("")
PROCESSED_DATASET = REPORT_PATH / "dataset_small_sample"
TRACKING_URI = "http://127.0.0.1:8080"

EXPERIMENT_NAME = "feature_scaling_determ"

TRAIN_PERCENTAGE = 0.7
EPOCHS_NUM = 100

STUDY_URI = "mlflow-artifacts:/873437386923880779/0dc152759bf74cd38342f5496397a038/artifacts/study.pkl"


def l2_norm(data):
    scaled_data = data.clone()
    scaled_data[TORCH.state_vertex].x = torch.nn.functional.normalize(
        data[TORCH.state_vertex].x, dim=0, p=2
    )
    scaled_data[TORCH.game_vertex].x[:, 1] = torch.nn.functional.normalize(
        data[TORCH.game_vertex].x[:, 1], dim=0, p=2
    )
    return scaled_data


def l_inf_norm(data):
    scaled_data = data.clone()
    scaled_data[TORCH.state_vertex].x = data[TORCH.state_vertex].x / (
        torch.max(data[TORCH.state_vertex].x, dim=0).values + 1e-12
    )
    scaled_data[TORCH.game_vertex].x[:, 1] = data[TORCH.game_vertex].x[:, 1] / (
        torch.max(data[TORCH.game_vertex].x[:, 1]) + 1e-12
    )
    return scaled_data


def min_max_scaling(data):
    scaled_data = data.clone()
    scaled_data[TORCH.state_vertex].x = (
        data[TORCH.state_vertex].x - torch.min(data[TORCH.state_vertex].x, dim=0).values
    ) / (
        torch.max(data[TORCH.state_vertex].x, dim=0).values
        - torch.min(data[TORCH.state_vertex].x, dim=0).values
        + 1e-12
    )
    scaled_data[TORCH.game_vertex].x[:, 1] = (
        data[TORCH.game_vertex].x[:, 1] - torch.min(data[TORCH.game_vertex].x[:, 1])
    ) / (
        torch.max(data[TORCH.game_vertex].x[:, 1])
        - torch.min(data[TORCH.game_vertex].x[:, 1])
        + 1e-12
    )
    return scaled_data


def main():
    print(GeneralConfig.DEVICE)

    mlflow.set_tracking_uri(uri=TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    torch.manual_seed(666)
    random.seed(9474)
    np.random.seed(637)
    torch.use_deterministic_algorithms(True, warn_only=True)

    transform_functions = [l_inf_norm, min_max_scaling]
    for transform_function in transform_functions:
        dataset = TrainingDataset(
            transform_function,
            RAW_DATASET,
            PROCESSED_DATASET,
            TRAIN_PERCENTAGE,
            load_to_cpu=True,
        )

        downloaded_artifact_path = mlflow.artifacts.download_artifacts(
            STUDY_URI, dst_path=str(REPORT_PATH)
        )
        study: optuna.Study = joblib.load(downloaded_artifact_path)

        def criterion_init():
            return nn.KLDivLoss(reduction="batchmean")

        def validate(model, dataset):
            criterion = criterion_init()
            result = validate_loss(
                model,
                dataset,
                criterion,
                200,
            )
            metric_name = str(criterion).replace("(", "_").replace(")", dataset.mode)
            metrics = {metric_name: result}
            return result, metrics

        if transform_function:
            run_name = f"{transform_function.__name__}_all_vertex_features"
        else:
            run_name = "baseline_all_vertex_feature"
        objective(
            run_name,
            study.best_trial,
            dataset,
            dynamic_dataset=False,
            model_init=lambda **model_params: StateModelEncoder(**model_params),
            criterion_init=lambda: nn.KLDivLoss(reduction="batchmean"),
            epochs=EPOCHS_NUM,
            validate=validate,
        )


if __name__ == "__main__":
    main()
