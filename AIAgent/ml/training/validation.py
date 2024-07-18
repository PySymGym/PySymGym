import multiprocessing as mp
from functools import wraps
from typing import Callable

import numpy as np
import torch
import tqdm
from common.errors import GameErrors
from epochs_statistics import StatisticsCollector
from config import GeneralConfig
from ml.inference import infer
from ml.play_game import play_game
from ml.training.dataset import TrainingDataset
from ml.training.wrapper import TrainingModelWrapper
from torch_geometric.loader import DataLoader


def catch_return_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e

    return wrapper


@catch_return_exception
def play_game_task(task):
    maps, dataset, wrapper = task

    result = play_game(
        with_predictor=wrapper,
        max_steps=GeneralConfig.MAX_STEPS,
        maps=maps,
        with_dataset=dataset,
    )
    torch.cuda.empty_cache()
    return result


def validate_coverage(
    statistics_collector: StatisticsCollector,
    model: torch.nn.Module,
    dataset: TrainingDataset,
    server_count: int,
    progress_bar_colour: str = "#ed95ce",
):
    """
    Evaluate model using symbolic execution engine. It runs in parallel.

    Parameters
    ----------
    statistics_collector: StatisticsCollector
        Collects statistics of training session.
    model : torch.nn.Module
        Model to evaluate
    dataset : TrainingDataset
        Dataset object for validation.
    server_count: int
        The number of game servers running in parallel.
    progress_bar_colour : str
        Your favorite colour for progress bar.
    """
    wrapper = TrainingModelWrapper(model)
    tasks = [([game_map], dataset, wrapper) for game_map in dataset.maps]
    with mp.Pool(server_count) as p:
        all_results = []
        for result in tqdm.tqdm(
            p.imap_unordered(play_game_task, tasks, chunksize=1),
            desc="validation",
            total=len(tasks),
            ncols=100,
            colour=progress_bar_colour,
        ):
            if isinstance(result, GameErrors):
                statistics_collector.fail(result.maps)
            else:
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
    dataloader = DataLoader(dataset, batch_size=2000)
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
