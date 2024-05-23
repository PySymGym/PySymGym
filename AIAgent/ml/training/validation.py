import multiprocessing as mp
from functools import partial
from multiprocessing.managers import AutoProxy
from typing import Callable

import numpy as np
import torch
import tqdm
from connection.broker_conn.classes import SVMInfo
from epochs_statistics import StatisticsCollector
from config import GeneralConfig
from epochs_statistics import StatisticsCollector
from ml.inference import infer
from ml.play_game import play_game
from ml.training.dataset import TrainingDataset
from ml.training.wrapper import TrainingModelWrapper
from torch_geometric.loader import DataLoader


def play_game_task(task):
    maps, dataset, wrapper = task[0], task[1], task[2]
    result = play_game(
        with_predictor=wrapper,
        max_steps=GeneralConfig.MAX_STEPS,
        maps=maps,
        with_dataset=dataset,
    )
    torch.cuda.empty_cache()
    return result


def play_game_task_exn_catcher(task):
    try:
        return play_game_task(task)
    except Exception as e:
        return e


def validate_coverage(
    statistics_collector: StatisticsCollector,
    model: torch.nn.Module,
    dataset: TrainingDataset,
    progress_bar_colour: str = "#ed95ce",
    server_count: int = 1,
):
    """
    Evaluate model using symbolic execution engine. It runs in parallel.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate
    dataset : TrainingDataset
        Dataset object for validation.
    progress_bar_colour : str
        Your favorite colour for progress bar.
    """
    wrapper = TrainingModelWrapper(model)
    tasks = [([game_map], dataset, wrapper) for game_map in dataset.maps]
    error = None

    with mp.Pool(server_count) as p:
        all_results = []
        for result in tqdm.tqdm(
            p.imap_unordered(play_game_task_exn_catcher, tasks, chunksize=1),
            desc="validation",
            total=len(tasks),
            ncols=100,
            colour=progress_bar_colour,
        ):
            if isinstance(result, Exception):
                error = result  # it is not possible to raise an exception here or to terminate pool due to the pool hanging
            else:
                all_results.extend(result)
    if isinstance(error, Exception):
        raise error

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
    statistics_collector.update_results(average_result, all_results)
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
        out = infer(model, batch)
        loss: torch.Tensor = criterion(out, batch.y_true)
        epoch_loss.append(loss.item())
    result = np.average(epoch_loss)
    print(f"Epoch {epoch}: {result}")
    return result
