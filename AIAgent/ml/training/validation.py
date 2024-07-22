import multiprocessing as mp
from functools import wraps
from typing import Callable

import numpy as np
import torch
import tqdm
import mlflow
from ml.training.epochs_statistics import StatisticsCollector
from common.classes import Map2Result
from common.errors import GameErrors
from config import GeneralConfig
from ml.inference import infer
from ml.play_game import play_game
from ml.training.dataset import TrainingDataset
from ml.training.wrapper import TrainingModelWrapper
from torch_geometric.loader import DataLoader
from paths import CURRENT_TABLE_PATH


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
    model: torch.nn.Module,
    dataset: TrainingDataset,
    epoch: int,
    server_count: int,
    progress_bar_colour: str = "#ed95ce",
):
    """
    Evaluate model using symbolic execution engine. It runs in parallel.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    dataset : TrainingDataset
        Dataset object for validation.
    epoch : int
        Epoch's number.
    server_count: int
        The number of game servers running in parallel.
    progress_bar_colour : str
        Your favorite colour for progress bar.
    """
    wrapper = TrainingModelWrapper(model)
    tasks = [([game_map], dataset, wrapper) for game_map in dataset.maps]
    statistics_collector = StatisticsCollector(CURRENT_TABLE_PATH)
    with mp.Pool(server_count) as p:
        all_results: list[Map2Result] = list()
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

    def avg_coverage(results, path_to_coverage: str) -> int:
        coverage = np.average(
            list(map(lambda result: getattr(result, path_to_coverage), results))
        )
        return coverage

    average_result = avg_coverage(
        list(map(lambda map2result: map2result.game_result, result)),
        "actual_coverage_percent",
    )
    statistics_collector.update_results(average_result, all_results)
    mlflow.log_metrics(
        {
            "average_dataset_state_result": avg_coverage(
                dataset.maps_results.values(), "coverage_percent"
            ),
            "average_result": average_result,
        },
        step=epoch,
    )
    mlflow.log_artifact(CURRENT_TABLE_PATH, str(epoch))

    return average_result, statistics_collector.get_failed_maps()


def validate_loss(
    model: torch.nn.Module,
    dataset: TrainingDataset,
    epoch: int,
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
    metric_name = str(criterion).replace("(", "_").replace(")", "_")
    mlflow.log_metric(metric_name, result, step=epoch)
    return result
