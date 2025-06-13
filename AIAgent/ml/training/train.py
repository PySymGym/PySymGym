from typing import Callable

import torch
import tqdm
import os
from torch_geometric.loader import DataLoader

from config import GeneralConfig
from ml.inference import infer


def train(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for batch in tqdm.tqdm(
        dataloader,
        desc="training",
        ncols=100,
        colour="#5c6fdb",
    ):
        batch.to(GeneralConfig.DEVICE)
        optimizer.zero_grad()
        out = infer(model, batch)
        loss = criterion(out, batch.y_true)
        if loss != 0:
            loss.backward()
            optimizer.step()
        del out
        del batch
    del dataloader
