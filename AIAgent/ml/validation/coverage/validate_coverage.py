import multiprocessing as mp
from typing import Optional

import torch
import tqdm
from common.classes import GameResult, Map2Result
from common.config.validation_config import SVMValidation, SVMValidationSendEachStep
from common.game import GameMap2SVM
from ml.dataset import Result, TrainingDataset
from ml.training.wrapper import TrainingModelWrapper
from ml.validation.coverage.game_managers.base_game_manager import BaseGameManager
from ml.validation.coverage.game_managers.each_step.each_step_game_manager import (
    EachStepGameManager,
)
from ml.validation.coverage.validate_coverage_utils import catch_return_exception


class ValidationCoverage:
    """
    Performs coverage validation of a model using symbolic execution. This class manages the parallel execution
    of game simulations across multiple maps and updates a provided dataset with the results.

    Attributes:
        model (`torch.nn.Module`): The model to be validated.
        dataset (`Optional[TrainingDataset]`): The dataset to update with validation results. Can be `None` if dataset update is not required.
    """

    def __init__(self, model: torch.nn.Module, dataset: Optional[TrainingDataset]):
        self.model = model
        self.dataset = dataset
        self._game_manager: Optional[BaseGameManager] = None

    def _evaluate_game_map(
        self,
        game_map2svm: GameMap2SVM,
    ) -> Map2Result | Exception:
        if self._game_manager is None:
            raise RuntimeError("Game manager has not been initialized yet")
        catching_play_game_map = catch_return_exception(
            self._game_manager.play_game_map
        )
        result: Map2Result | Exception = catching_play_game_map(game_map2svm)
        if (
            self.dataset is not None
            and isinstance(result, Map2Result)
            and isinstance(result.game_result, GameResult)
        ):
            game_result = result.game_result
            map_result = Result(
                coverage_percent=game_result.actual_coverage_percent,
                negative_tests_number=-game_result.tests_count,
                negative_steps_number=-game_result.steps_count,
                errors_number=game_result.errors_count,
            )
            steps = self._game_manager.get_game_steps(game_map2svm.GameMap)
            self.dataset.update_map(game_map2svm.GameMap.MapName, map_result, steps)
            del steps
        return result

    def validate_coverage(
        self,
        maps: list[GameMap2SVM],
        validation_config: SVMValidation,
        progress_bar_colour: str = "#ed95ce",
    ):
        """
        Evaluate model using symbolic execution engine. It runs in parallel.

        Parameters
        ----------
        maps : list[GameMap2SVM]
            List of maps description.
        validation_svm : SVMValidation
            Validation info.
        progress_bar_colour : str
            Your favorite colour for progress bar.
        """
        self._game_manager = self._get_game_manager(validation_config)
        with mp.Pool(validation_config.process_count) as p:
            all_results: list[Map2Result | Exception] = list()
            for result in tqdm.tqdm(
                p.imap_unordered(self._evaluate_game_map, maps, chunksize=1),
                desc="validation",
                total=len(maps),
                ncols=100,
                colour=progress_bar_colour,
            ):
                all_results.append(result)
        return all_results

    def _get_game_manager(self, validation_config: SVMValidation) -> BaseGameManager:
        if isinstance(validation_config, SVMValidationSendEachStep):
            return EachStepGameManager(TrainingModelWrapper(self.model))
        raise RuntimeError(f"There is no game manager suitable to {validation_config}")
