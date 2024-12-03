import multiprocessing as mp

import torch
import tqdm
from common.classes import Map2Result, GameMap2SVM
from common.config import ValidationSVMViaServer
from ml.training.dataset import TrainingDataset
from ml.training.validation.validate_coverage_utils import play_game_task
from ml.training.wrapper import TrainingModelWrapper


def validate_coverage_via_server(
    model: torch.nn.Module,
    dataset: TrainingDataset,
    maps: list[GameMap2SVM],
    validation_config: ValidationSVMViaServer,
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
    validation_config : ValidationSVMViaServer
        Validation config from the config file.
    progress_bar_colour : str
        Your favorite colour for progress bar.
    """
    wrapper = TrainingModelWrapper(model)
    tasks = [(game_map2svm, dataset, wrapper) for game_map2svm in maps]
    with mp.Pool(validation_config.servers_count) as p:
        all_results: list[Map2Result] = list()
        for result in tqdm.tqdm(
            p.imap_unordered(play_game_task, tasks, chunksize=1),
            desc="validation",
            total=len(tasks),
            ncols=100,
            colour=progress_bar_colour,
        ):
            all_results.append(result)
    return all_results
