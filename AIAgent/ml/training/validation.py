import multiprocessing as mp
from functools import wraps
from typing import Callable

import numpy as np
import torch
import tqdm
from common.config import ValidationWithSVMs
from common.classes import Map2Result, GameMap2SVM
from config import GeneralConfig
from ml.game.errors_game import GameError
from ml.inference import infer
from ml.game.play_game import play_game
from ml.training.dataset import TrainingDataset
from ml.training.wrapper import TrainingModelWrapper
from ml.training.statistics import get_svms_statistics

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
def play_game_task(
    task: tuple[GameMap2SVM, TrainingDataset, TrainingModelWrapper],
) -> Map2Result:
    game_map2svm, dataset, wrapper = task
    result = play_game(
        with_predictor=wrapper,
        max_steps=GeneralConfig.MAX_STEPS,
        game_map2svm=game_map2svm,
        with_dataset=dataset,
    )
    torch.cuda.empty_cache()
    return result


def validate_coverage(
    model: torch.nn.Module,
    dataset: TrainingDataset,
    maps: list[GameMap2SVM],
    validation_config: ValidationWithSVMs,
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
    maps : list[GameMap2SVM]
        List of maps description.
    validation_config : ValidationWithSVMs
        Validation config from the config file.
    progress_bar_colour : str
        Your favorite colour for progress bar.
    """
    wrapper = TrainingModelWrapper(model)
    tasks = [(game_map2svm, dataset, wrapper) for game_map2svm in maps]
    with mp.Pool(validation_config.servers_count) as p:
        all_results: list[Map2Result] = list()
        maps_to_remove: list[GameMap2SVM] = list()
        for result in tqdm.tqdm(
            p.imap_unordered(play_game_task, tasks, chunksize=1),
            desc="validation",
            total=len(tasks),
            ncols=100,
            colour=progress_bar_colour,
        ):
            if isinstance(result, GameError):
                need_to_save_map: bool = result.need_to_save_map()
                if not need_to_save_map:
                    maps_to_remove.append(result.map2result.map)
                result = result.map2result
            all_results.append(result)

    result, metrics = get_svms_statistics(all_results, validation_config, dataset)
    return result, metrics, maps_to_remove


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
    metric_name = str(criterion).replace("(", "_").replace(")", "_")
    metrics = {metric_name: result}
    return result, metrics
