import multiprocessing as mp
import os
from functools import partial
from multiprocessing.managers import AutoProxy
from typing import Callable

import numpy as np
import torch
import tqdm
from common.classes import SVMInfo
from config import GeneralConfig
from epochs_statistics.classes import StatisticsCollector
from epochs_statistics.tables import create_pivot_table
from ml.inference import infer
from ml.play_game import play_game
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
    # table, _, _ = create_pivot_table(
    #     svm_info.name,
    #     {wrapper: sorted(all_results, key=lambda x: x.map.MapName)},
    # )
    new_table = convert_to_des(all_results)
    # table.to_csv("TARGET_PD.csv")
    new_table.to_csv("NEW_PD.csv")
    # assert False
    statistics_collector.update_results(epoch, svm_info.name, average_result, new_table)
    return average_result


import pandas as pd


def convert_to_des(map2result_list):
    result_list = []
    for map2result in map2result_list:
        map_name = map2result.map.MapName
        game_result_str = map2result.game_result.printable(verbose=True)
        result_list.append((map_name, game_result_str))

    df = pd.DataFrame(result_list, columns=["MapName", "GameResult"])

    return df


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
        out = infer(model, batch)
        loss: torch.Tensor = criterion(out, batch.y_true)
        epoch_loss.append(loss.item())
    result = np.average(epoch_loss)
    print(f"Epoch {epoch}: {result}")
    return result
