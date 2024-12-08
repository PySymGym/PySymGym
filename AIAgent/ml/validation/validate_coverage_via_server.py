import multiprocessing as mp

import torch
import tqdm
from common.classes import GameMap2SVM, Map2Result
from common.config.validation_config import ValidationSVMViaServer
from ml.dataset import TrainingDataset
from ml.training.wrapper import TrainingModelWrapper
from ml.validation.game.play_game_via_server import play_game
from ml.validation.validate_coverage_utils import catch_return_exception


@catch_return_exception
def play_game_task(
    task: tuple[GameMap2SVM, TrainingDataset, TrainingModelWrapper],
) -> Map2Result:
    game_map2svm, dataset, wrapper = task
    result = play_game(
        with_predictor=wrapper,
        game_map2svm=game_map2svm,
        with_dataset=dataset,
    )
    torch.cuda.empty_cache()
    return result


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
    with mp.Pool(validation_config.process_count) as p:
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
