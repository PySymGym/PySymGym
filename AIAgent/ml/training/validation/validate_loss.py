from typing import Callable

import numpy as np
import torch
import tqdm
from config import GeneralConfig
from ml.inference import infer
from ml.training.dataset import TrainingDataset
from torch_geometric.loader import DataLoader


def validate_loss(
    model: torch.nn.Module,
    dataset: TrainingDataset,
    criterion: Callable,
    batch_size: int,
    progress_bar_colour: str = "#975cdb",
):
    epoch_loss = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in tqdm.tqdm(
        dataloader, desc="test", ncols=100, colour=progress_bar_colour
    ):
        batch.to(GeneralConfig.DEVICE)
        out = infer(model, batch)
        loss: torch.Tensor = criterion(out, batch.y_true)
        epoch_loss.append(loss.item())
    result = np.average(epoch_loss)
    return result
