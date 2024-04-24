from typing import Callable
import torch
import tqdm
from config import GeneralConfig

from torch_geometric.loader import DataLoader


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
        out = model(
            game_x=batch["game_vertex"].x,
            state_x=batch["state_vertex"].x,
            edge_index_v_v=batch["game_vertex", "to", "game_vertex"].edge_index,
            edge_type_v_v=batch["game_vertex", "to", "game_vertex"].edge_type,
            edge_index_history_v_s=batch[
                "game_vertex", "history", "state_vertex"
            ].edge_index,
            edge_attr_history_v_s=batch[
                "game_vertex", "history", "state_vertex"
            ].edge_attr,
            edge_index_in_v_s=batch["game_vertex", "in", "state_vertex"].edge_index,
            edge_index_s_s=batch[
                "state_vertex", "parent_of", "state_vertex"
            ].edge_index,
        )
        loss = criterion(out, batch.y_true)
        if loss != 0:
            loss.backward()
            optimizer.step()
        del out
        del batch
    del dataloader
