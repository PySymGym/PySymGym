import os

from ml.het_gnn_test_train import HetGNNTestTrain
from ml.models.RGCNEdgeTypeTAG3VerticesDoubleHistory2.model import StateModelEncoder
from ml.training.dataset import TrainingDataset
from paths import RAW_DATASET_PATH

DATASET_PATH = RAW_DATASET_PATH / "pretrain_dataset"
if not DATASET_PATH.exists():
    os.makedirs(DATASET_PATH)

if __name__ == "__main__":
    dataset = TrainingDataset(RAW_DATASET_PATH, DATASET_PATH)
    pr = HetGNNTestTrain(StateModelEncoder, 64)
    pr.train_and_save(DATASET_PATH, 20, "./ml/models/")
