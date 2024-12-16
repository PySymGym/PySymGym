from typing import Callable

import torch
import tqdm
from torch_geometric.loader import DataLoader

from config import GeneralConfig
from ml.inference import infer


def train(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
):
    for batch in tqdm.tqdm(
        dataloader,
        desc="training",
        ncols=100,
        colour="#5c6fdb",
    ):
        batch.to(GeneralConfig.DEVICE)
        optimizer.zero_grad()
        out = infer(model, batch)
        batch.y_true[batch.y_true > 0] = 1
        loss = criterion(out, batch.y_true)
        if loss != 0:
            loss.backward()
            optimizer.step()
        del out
        del batch
    del dataloader
