from pathlib import Path

from ml.dataset import TrainingDataset
from ml.validation.validate_loss import validate_loss
from config import GeneralConfig
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2Parametrized.model import (
    StateModelEncoder,
)
import mlflow
import torch
from torch import nn
from run_training import objective
import numpy as np
import random
from ml.inference import TORCH

RAW_DATASET = Path("")
PROCESSED_DATASET = "./report/dataset"
TRACKING_URI = "http://127.0.0.1:8080"

EXPERIMENT_NAME = "history-edges-normalization"

TRAIN_PERCENTAGE = 0.7
EPOCHS_NUM = 100

def l2_norm(data):
    scaled_data = data.clone()
    attr = data[TORCH.gamevertex_history_statevertex].edge_attr.to(torch.float)
    scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = (
        torch.nn.functional.normalize(attr, dim=0, p=2)
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
    ) / (torch.max(attr) - torch.min(attr) + 1e-12)
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
    scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = (
        edge_attr - median
    ) / iqr
    return scaled_data


def reciprocal_norm(data):
    scaled_data = data.clone()
    edge_attr = data[TORCH.gamevertex_history_statevertex].edge_attr.to(torch.float)

    epsilon = 1e-12
    edge_attr_normalized = 1 - 1 / (edge_attr + epsilon)

    scaled_data[TORCH.gamevertex_history_statevertex].edge_attr = edge_attr_normalized
    return scaled_data


def main():
    print(GeneralConfig.DEVICE)

    mlflow.set_tracking_uri(uri=TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    normalization_functions = [
        None,
        l2_norm
    ]

    for normalization_function in normalization_functions:
        dataset = TrainingDataset(
            normalization_function,
            RAW_DATASET,
            PROCESSED_DATASET,
            TRAIN_PERCENTAGE,
            load_to_cpu=True,
        )

        def criterion_init():
            return nn.KLDivLoss(reduction="batchmean")

        def validate(model, dataset):
            criterion = criterion_init()
            result = validate_loss(
                model,
                dataset,
                criterion,
                100,
            )
            metric_name = str(criterion).replace("(", "_").replace(")", dataset.mode)
            metrics = {metric_name: result}
            return result, metrics

        if normalization_function:
            run_name = f"{normalization_function.__name__}"
        else:
            run_name = "None"
        objective(
            run_name,
            dataset,
            dynamic_dataset=False,
            model_init=lambda **model_params: StateModelEncoder(**model_params),
            criterion_init=lambda: nn.KLDivLoss(reduction="batchmean"),
            epochs=EPOCHS_NUM,
            validate=validate,
        )


if __name__ == "__main__":
    main()
