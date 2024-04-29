from functools import partial
import multiprocessing as mp
from typing import Callable
from multiprocessing.managers import AutoProxy

import numpy as np
import torch
import tqdm
import os
from common.classes import SVMInfo
from config import GeneralConfig
from epochs_statistics.tables import create_pivot_table
from epochs_statistics.classes import StatisticsCollector
from learning.play_game import play_game
from ml.training.dataset import TrainingDataset
from ml.training.wrapper import TrainingModelWrapper
from torch_geometric.loader import DataLoader


def play_game_task(svm_info: SVMInfo, task):
    maps, dataset, wrapper = task[0], task[1], task[2]
    result = play_game(
        svm_info=svm_info,
        with_predictor=wrapper,
        max_steps=GeneralConfig.MAX_STEPS,
        maps=maps,
        with_dataset=dataset,
    )
    torch.cuda.empty_cache()
    return result


def validate_coverage(
    svm_info: SVMInfo,
    statistics_collector: "AutoProxy[StatisticsCollector]",
    model: torch.nn.Module,
    epoch: int,
    dataset: TrainingDataset,
    progress_bar_colour: str = "#ed95ce",
):
    """
    Evaluate model using symbolic execution engine. It runs in parallel.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate
    epoch : int
        Epoch number to write result.
    dataset : TrainingDataset
        Dataset object for validation.
    progress_bar_colour : str
        Your favorite colour for progress bar.
    """
    wrapper = TrainingModelWrapper(model)
    tasks = [([game_map], dataset, wrapper) for game_map in dataset.maps]
    server_count = svm_info.count
    with mp.Pool(server_count) as p:
        all_results = []
        for result in tqdm.tqdm(
            p.imap_unordered(partial(play_game_task, svm_info), tasks, chunksize=1),
            desc="validation",
            total=len(tasks),
            ncols=100,
            colour=progress_bar_colour,
        ):
            all_results.extend(result)

    print(
        "Average dataset state result",
        np.average(
            list(
                map(
                    lambda dataset_map_result: dataset_map_result[0],
                    dataset.maps_results.values(),
                )
            )
        ),
    )
    average_result = np.average(
        list(
            map(
                lambda map_result: map_result.game_result.actual_coverage_percent,
                all_results,
            )
        )
    )
    table, _, _ = create_pivot_table(
        svm_info.name,
        {wrapper: sorted(all_results, key=lambda x: x.map.MapName)},
    )
    statistics_collector.update_results(epoch, svm_info.name, average_result, table)
    return average_result


def validate_loss(
    model: torch.nn.Module,
    epoch: int,
    dataset: TrainingDataset,
    criterion: Callable,
    progress_bar_colour: str = "#975cdb",
):
    epoch_loss = []
    dataloader = DataLoader(dataset, 1)
    for batch in tqdm.tqdm(
        dataloader, desc="test", ncols=100, colour=progress_bar_colour
    ):
        batch.to(GeneralConfig.DEVICE)
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
        loss: torch.Tensor = criterion(out, batch.y_true)
        epoch_loss.append(loss.item())
    result = np.average(epoch_loss)
    print(f"Epoch {epoch}: {result}")
    return result
