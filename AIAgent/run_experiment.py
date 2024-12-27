from pathlib import Path

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
import numpy as np
import random

RAW_DATASET = Path("")
PROCESSED_DATASET = "./AIAgent/report/dataset"
TRACKING_URI = "http://127.0.0.1:8080"

EXPERIMENT_NAME = "history-edges-normalization"

TRAIN_PERCENTAGE = 0.7
EPOCHS_NUM = 100


def main():
    print(GeneralConfig.DEVICE)

    mlflow.set_tracking_uri(uri=TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    dataset = TrainingDataset(
        None,
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

    objective(
        "None",
        dataset,
        dynamic_dataset=False,
        model_init=lambda **model_params: StateModelEncoder(**model_params),
        criterion_init=lambda: nn.KLDivLoss(reduction="batchmean"),
        epochs=EPOCHS_NUM,
        validate=validate,
    )


if __name__ == "__main__":
    main()
