import glob
import json
import logging
import os
import subprocess
import traceback
from multiprocessing.managers import Namespace
from pathlib import Path

import torch
from common.classes import GameFailed, GameResult, Map2Result
from common.game import GameMap, GameMap2SVM, GameState
from func_timeout import FunctionTimedOut
from ml.dataset import convert_input_to_tensor
from ml.validation.coverage.game_managers.base_game_manager import (
    BaseGameManager,
    BaseGamePreparator,
)
from ml.validation.coverage.game_managers.utils import set_timeout_if_needed
from onyx import load_gamestate
from paths import CURRENT_MODEL_PATH, MODEL_KWARGS_PATH, REPORT_PATH
from torch_geometric.data.hetero_data import HeteroData

# svm substituation variables
MAP_NAME = "MapName"
MODEL_PATH = "ModelPath"
STEPS_TO_PLAY = "StepsToPlay"
DEFAULT_SEARCHER = "DefaultSearcher"
STEPS_TO_START = "StepsToStart"
ASSEMBLY_FULL_NAME = "AssemblyFullName"
NAME_OF_OBJECT_TO_COVER = "NameOfObjectToCover"
OUTPUT_DIR = "OutputDir"

CURRENT_ONNX_MODEL_PATH = REPORT_PATH / "model.onnx"
GAMESTATE_EXAMPLE_PATH = "../resources/onnx/reference_gamestates/7825_gameState.json"
SVMS_OUTPUT_PATH = REPORT_PATH / "svms_output"


class ModelGamePreparator(BaseGamePreparator):
    def __init__(
        self,
        namespace: Namespace,
        model: torch.nn.Module,
        path_to_model: str = CURRENT_MODEL_PATH,
    ):
        self._model = model
        self._path_to_model = path_to_model
        super().__init__(namespace)

    def _prepare(self):
        import_model_fqn = self._model.__class__.__module__ + ".StateModelEncoder"
        launch_command = f"""python onyx.py --sample-gamestate {GAMESTATE_EXAMPLE_PATH}
                            --pytorch-model {self._path_to_model}
                            --savepath {CURRENT_ONNX_MODEL_PATH}
                            --import-model-fqn {import_model_fqn}
                            --model-kwargs {MODEL_KWARGS_PATH}"""
        logging.info(f"Launch command: {launch_command}")
        proc = subprocess.Popen(
            launch_command.split(),
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        proc.wait()
        assert proc.returncode == 0


class ModelGameManager(BaseGameManager):
    GAME_STATE_SUFFIX = "gameState"
    NN_OUTPUT_SUFFIX = "nn_output"

    def __init__(
        self,
        namespace: Namespace,
        model: torch.nn.Module,
        path_to_model: str = CURRENT_MODEL_PATH,
    ):
        self._namespace = namespace
        self._model = model
        self._path_to_model = path_to_model
        super().__init__(self._namespace)

    def _get_output_dir(self, game_map: GameMap) -> Path:
        return SVMS_OUTPUT_PATH / f"{game_map.MapName}"

    def _run_game_process(self, game_map2svm: GameMap2SVM):
        game_map, svm_info = game_map2svm.GameMap, game_map2svm.SVMInfo
        launch_command = svm_info.launch_command.format(
            **{
                STEPS_TO_PLAY: game_map.StepsToPlay,
                DEFAULT_SEARCHER: game_map.DefaultSearcher,
                STEPS_TO_START: game_map.StepsToStart,
                ASSEMBLY_FULL_NAME: game_map.AssemblyFullName,
                NAME_OF_OBJECT_TO_COVER: game_map.NameOfObjectToCover,
                MAP_NAME: game_map.MapName,
                MODEL_PATH: Path(CURRENT_ONNX_MODEL_PATH).absolute(),
                OUTPUT_DIR: self._get_output_dir(game_map2svm.GameMap).absolute(),
            },
        )
        logging.info(f"Launch command: {launch_command}")
        proc = subprocess.Popen(
            launch_command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(svm_info.server_working_dir),
        )
        return proc

    def _kill_game_process(self, proc: subprocess.Popen):
        proc.kill()
        logging.warning(f"Process {proc.pid} was intentionally killed")

    @set_timeout_if_needed
    def _wait_for_game_over(self, proc: subprocess.Popen) -> None:
        out, _ = proc.communicate()
        logging.debug(out)

    def _get_result(self, game_map2svm: GameMap2SVM) -> GameResult | GameFailed:
        output_dir = self._get_output_dir(game_map2svm.GameMap)
        result_file = (
            output_dir / f"{game_map2svm.GameMap.MapName}result"
        )  # TODO: the path to result must be documented
        try:
            with open(result_file) as f:
                actual_coverage_percent, tests_count, steps_count, errors_count = (
                    f.read().split()
                )
                actual_coverage_percent = int(actual_coverage_percent)
                tests_count = int(tests_count)
                steps_count = int(steps_count)
                errors_count = int(errors_count)
                result = GameResult(
                    steps_count=steps_count,
                    tests_count=tests_count,
                    errors_count=errors_count,
                    actual_coverage_percent=actual_coverage_percent,
                )
                logging.info(
                    f"Finished map {game_map2svm.GameMap.MapName} "
                    f"in {result.steps_count} steps,"
                    f"actual coverage: {result.actual_coverage_percent:.2f}"
                )
        except FileNotFoundError:
            logging.error(
                f"The result of {str(game_map2svm.GameMap.MapName)} cannot be found after completing the game?!"
            )
            result = GameFailed("There is no file with game result")
        return result

    def _play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        proc = self._run_game_process(game_map2svm)
        try:
            _ = self._wait_for_game_over(proc)
            game_result = self._get_result(game_map2svm)
        except FunctionTimedOut as error:
            logging.warning(
                f"Timeouted on map {game_map2svm.GameMap.MapName} with {error.timedOutAfter}s"
            )
            self._kill_game_process(proc)
            game_result = GameFailed(reason=type(error))
        except Exception as e:
            trace = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
            logging.error(trace)
            self._kill_game_process(proc)
            game_result = GameFailed(reason=type(e))

        return Map2Result(game_map2svm, game_result)

    def get_game_steps(self, game_map: GameMap) -> list[HeteroData]:
        output_dir = self._get_output_dir(game_map)
        files = [
            f
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]
        states = dict()
        nn_outputs = dict()

        def get_index(s: str) -> int:
            return int(s[0 : s.find("_")])

        def get_game_state(path: str) -> GameState:
            with open(path, "r") as gamestate_file:
                return load_gamestate(gamestate_file)

        def get_nn_output(path: str):
            with open(path, "r") as file:
                return json.loads(file.read())

        for file in files:
            if file.endswith(ModelGameManager.GAME_STATE_SUFFIX) or file.endswith(
                ModelGameManager.NN_OUTPUT_SUFFIX
            ):
                index = get_index(file)
                full_path = output_dir / file
                if file.endswith(ModelGameManager.GAME_STATE_SUFFIX):
                    states[index] = get_game_state(full_path)
                else:
                    nn_outputs[index] = get_nn_output(full_path)
        map_steps = []
        for state, nn_output in zip(
            dict(sorted(states.items())).values(),
            dict(sorted(nn_outputs.items())).values(),
        ):
            hetero_input, _ = convert_input_to_tensor(state)
            nn_output = list(map(lambda x: [x], nn_output[0]))
            hetero_input["y_true"] = torch.Tensor(nn_output)
            map_steps.append(hetero_input)
        return map_steps

    def _create_preparator(self):
        return ModelGamePreparator(self._namespace, self._model, self._path_to_model)

    def delete_game_steps(self, game_map: GameMap):
        def delete_files_with_suffix(suffix: str):
            directory = (
                Path(ModelGameManager.NN_OUTPUT_SUFFIX) / game_map.AssemblyFullName
            )
            try:
                pattern = os.path.join(directory, f"*{suffix}")
                files_to_delete = glob.glob(pattern, recursive=True)

                for file_path in files_to_delete:
                    os.remove(file_path)
                    logging.debug(f"File deleted: {file_path}")

            except Exception as e:
                logging.error(
                    "Something happened during deletion:"
                    + "\n".join(
                        traceback.format_exception(e.__class__, e, e.__traceback__)
                    )
                )

        delete_files_with_suffix(ModelGameManager.GAME_STATE_SUFFIX)
        delete_files_with_suffix(ModelGameManager.NN_OUTPUT_SUFFIX)
