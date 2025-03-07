import json
import logging
import socket
import subprocess
import time
from multiprocessing.managers import Namespace
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import yaml
from common.classes import GameFailed, GameResult, Map2Result
from common.file_system_utils import delete_dir
from common.game import GameMap, GameMap2SVM, GameState
from common.network_utils import next_free_port
from common.validation_coverage.svm_info import SVMInfo
from func_timeout import FunctionTimedOut
from ml.dataset import convert_input_to_tensor
from ml.validation.coverage.game_managers.base_game_manager import (
    BaseGameManager,
    BaseGamePreparator,
)
from ml.validation.coverage.game_managers.each_step.game_states_utils import (
    update_game_state,
)
from ml.validation.coverage.game_managers.model.classes import (
    GameFailedDetails,
    GameResultDetails,
    ModelGameMapInfo,
    ModelGameStep,
    SVMConnectionInfo,
)
from ml.validation.coverage.game_managers.utils import set_timeout_if_needed
from onyx import entrypoint, load_gamestate, resolve_import_model
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
PORT = "Port"

CURRENT_ONNX_MODEL_PATH = REPORT_PATH / "model.onnx"
GAMESTATE_EXAMPLE_PATH = "../resources/onnx/reference_gamestates/7825_gameState.json"
SVMS_OUTPUT_PATH = REPORT_PATH / "svms_output"

FULL_COVERAGE_PERCENT = 100
# 1. TODO: find a better place for it
# 2. TODO: remove magic consts in other modules

# TODO: consider unifying message logging

# TODO: docs


class ModelGamePreparator(BaseGamePreparator):
    def __init__(
        self,
        namespace: Namespace,
        model: torch.nn.Module,
        path_to_model: Path = CURRENT_MODEL_PATH,
    ):
        self._model = model
        self._path_to_model = path_to_model
        super().__init__(namespace)

    def _create_onnx_model(self):
        import_model_fqn = (
            f"{self._model.__class__.__module__}.{self._model.__class__.__name__}"
        )
        with open(MODEL_KWARGS_PATH, "r") as file:
            model_kwargs = yaml.safe_load(file)

        with open(GAMESTATE_EXAMPLE_PATH) as gamestate_file:
            entrypoint(
                sample_gamestate=load_gamestate(gamestate_file),
                pytorch_model_path=self._path_to_model,
                onnx_savepath=CURRENT_ONNX_MODEL_PATH,
                model_def=resolve_import_model(import_model_fqn),
                model_kwargs=model_kwargs,
            )

    def _clean_output_folder(self):
        svms_output_path = Path(SVMS_OUTPUT_PATH)
        if svms_output_path.exists():
            return delete_dir(SVMS_OUTPUT_PATH)

    def _prepare(self):
        self._create_onnx_model()
        self._clean_output_folder()


class ModelGameManager(BaseGameManager):
    def __init__(
        self,
        namespace: Namespace,
        model: torch.nn.Module,
        path_to_model: Path = CURRENT_MODEL_PATH,
    ):
        self._namespace = namespace
        self._shared_lock = namespace.shared_lock
        self._model = model
        self._path_to_model = path_to_model
        self._games_info: dict[str, ModelGameMapInfo] = {}
        super().__init__(self._namespace)

    def _play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        game_map = game_map2svm.GameMap
        map_name = game_map.MapName
        game_map_info = ModelGameMapInfo(
            total_game_state=None,
            total_steps=[],
            proc=None,  # type: ignore
            game_result=None,  # type: ignore
            # TODO: avoid ugly code
        )
        self._games_info[map_name] = game_map_info
        running_res = None
        try:
            # TODO: rewrite it more clear to read
            running_res = self._run_game_process(game_map2svm)
            proc, port, socket = running_res
            game_map_info.proc = proc
            svm_connection_info = SVMConnectionInfo(occupied_port=port, socket=socket)
            game_result = self._get_result(game_map2svm, proc)
            # TODO: There is confusion here because of the ambiguous name of the 'GameResult'. Renaming is needed.
            if isinstance(game_result, GameResult):
                game_result = GameResultDetails(
                    game_result=game_result,
                    svm_connection_info=svm_connection_info,
                )
            else:
                game_result = GameFailedDetails(game_failed=game_result)
        except FunctionTimedOut as error:
            logging.warning(
                f"Timeouted on map {map_name} with {error.timedOutAfter}s after {self._games_info[map_name].total_steps}"
            )
            self._kill_game_process(proc, logging.warning)
            game_failed = GameFailed(reason=type(error))
            game_result = GameFailedDetails(game_failed=game_failed)
        except Exception as e:
            logging.error(e, exc_info=True)
            if running_res is not None:
                self._kill_game_process(proc, logging.error)
            game_failed = GameFailed(reason=type(e))
            game_result = GameFailedDetails(game_failed=game_failed)
        finally:
            torch.cuda.empty_cache()
            game_map_info.game_result = game_result  # type: ignore
        if isinstance(game_result, GameResultDetails):  # type: ignore
            game_result = game_result.game_result
        else:
            game_result = game_result.game_failed  # type: ignore # TODO: Handle some weird type error
        return Map2Result(game_map2svm, game_result)

    def _get_output_dir(self, game_map: GameMap) -> Path:
        return SVMS_OUTPUT_PATH / f"{game_map.MapName}"

    def _look_for_free_port(
        self, svm_info: SVMInfo, attempts=100
    ) -> Tuple[int, socket.socket]:
        if attempts <= 0:
            raise RuntimeError("Failed to occupy port")
        logging.debug(f"Looking for port... Attempls left: {attempts}.")
        try:
            with self._shared_lock:
                port = next_free_port(svm_info.min_port, svm_info.max_port)
                logging.debug(f"Try to occupy {port=}")
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.bind(
                    ("localhost", port)
                )  # TODO: working within a local network
                server_socket.listen(1)
                return port, server_socket
        except OSError:
            logging.debug("Failed to occupy port")
            return self._look_for_free_port(svm_info, attempts - 1)

    def _run_game_process(self, game_map2svm: GameMap2SVM):
        game_map, svm_info = game_map2svm.GameMap, game_map2svm.SVMInfo

        port, server_socket = self._look_for_free_port(svm_info)
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
                PORT: port,
            },
        )
        logging.info(f"Launch command: {launch_command}")
        proc = subprocess.Popen(
            launch_command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(svm_info.server_working_dir).resolve(),
            encoding="utf-8",
        )
        return proc, port, server_socket

    def _kill_game_process(
        self, proc: subprocess.Popen, logger: Callable[..., None] = logging.debug
    ):
        proc.kill()
        _ = self._get_and_log_proc_output(proc, logger)
        logging.warning(f"Process {proc.pid} was intentionally killed")

    @set_timeout_if_needed
    def _get_result(
        self, game_map2svm: GameMap2SVM, proc: subprocess.Popen
    ) -> GameResult | GameFailed:
        game_map = game_map2svm.GameMap
        map_name = game_map.MapName
        output_dir = self._get_output_dir(game_map)
        result_file = (
            output_dir / f"{map_name}result"
        )  # TODO: the path to result must be documented
        while True:
            try:
                with open(result_file, "r") as f:
                    data = f.read().split()
                (actual_coverage_percent, tests_count, steps_count, errors_count) = map(
                    int, data
                )
                steps_count -= game_map.StepsToStart
                result = GameResult(
                    steps_count=steps_count,
                    tests_count=tests_count,
                    errors_count=errors_count,
                    actual_coverage_percent=actual_coverage_percent,
                )
                actual_coverage_formatted = f"{actual_coverage_percent:.2f}"
                if steps_count == 0:
                    if actual_coverage_percent != FULL_COVERAGE_PERCENT:
                        logging.warning(
                            f"There is neither any step chosen by oracle, nor full coverage achieved. Actual coverage: {actual_coverage_formatted}. Immediate GameOver on {map_name}."
                        )
                        return GameResult(game_map2svm.GameMap.StepsToPlay, 0, 0, 0)
                    else:
                        logging.warning(
                            f"The map {map_name} was completely covered without an oracle?!"
                        )
                elif (
                    steps_count < game_map.StepsToPlay
                    and actual_coverage_percent != FULL_COVERAGE_PERCENT
                ):
                    logging.warning(
                        f"Not all steps exhausted on {game_map.MapName} with non-100% coverage. Steps taken by oracle: {steps_count}, actual coverage: {actual_coverage_formatted}."
                    )
                logging.info(
                    f"Finished map {map_name} "
                    f"in {result.steps_count} steps,"
                    f"actual coverage: {result.actual_coverage_percent:.2f}"
                )
                return result
            except (FileNotFoundError, ValueError) as e:
                ret_code = proc.poll()
                if ret_code is None:
                    time.sleep(3)
                    continue
                if isinstance(e, ValueError):
                    msg = f"Incorrect result of {map_name}"
                else:
                    msg = f"The result of {map_name} cannot be found after completing the game?!"
                logging.error(msg)
                return GameFailed(msg)

    def get_game_steps(self, game_map: GameMap) -> Optional[list[HeteroData]]:
        game_map_info = self._games_info.get(game_map.MapName)
        return (
            game_map_info.total_steps
            if game_map_info and game_map_info.total_steps
            else None
        )

    def are_steps_required(self, game_map: GameMap, required: bool):
        map_name = game_map.MapName
        if map_name not in self._games_info:
            msg = f"Can't find game info of {game_map.MapName}"
            logging.warning(msg)
            return
        game_map_info = self._games_info[map_name]
        if isinstance(game_map_info.game_result, GameResultDetails):
            game_result = game_map_info.game_result
        else:
            self._get_and_log_proc_output(game_map_info.proc, logging.error)
            return

        def get_conn():
            svm_connection_info = game_result.svm_connection_info
            port = svm_connection_info.occupied_port
            logging.debug(f"Server listening on port {port}")
            conn, addr = svm_connection_info.socket.accept()
            logging.debug(f"Connection from {addr}")
            return conn

        def alert_svm_about_step_saving(conn: socket.socket, need_to_save_steps: bool):
            conn.sendall(bytes([1 if need_to_save_steps else 0]))
            conn.shutdown(socket.SHUT_WR)

        conn = get_conn()
        alert_svm_about_step_saving(conn, required)

        if (
            not game_map_info.total_steps and required
        ):  # there is no already taken steps

            def get_steps_from_svm() -> list:
                output_dir = self._get_output_dir(game_map)
                steps = output_dir / f"{map_name}_steps"
                game_map_info.proc.wait()
                with open(steps, "br") as f:
                    steps_json = json.load(f)
                steps = list(map(lambda v: ModelGameStep.from_dict(v), steps_json))  # type: ignore
                return steps

            def convert_steps_to_hetero(steps: list):
                if not steps:
                    return []
                step = steps.pop(0)
                game_state, nn_output = step.GameState, step.Output

                def get_hetero_data(game_state: GameState, nn_output: list):
                    hetero_input, _ = convert_input_to_tensor(game_state)
                    nn_output = [[x] for x in nn_output[0]]
                    hetero_input["y_true"] = torch.Tensor(nn_output)
                    return hetero_input

                steps_hetero = [get_hetero_data(game_state, nn_output)]
                for step in steps:
                    delta = step.GameState
                    game_state = update_game_state(game_state, delta)
                    hetero_input, _ = convert_input_to_tensor(game_state)
                    hetero_input = get_hetero_data(game_state, step.Output)
                    steps_hetero.append(hetero_input)
                return steps_hetero

            try:
                steps = get_steps_from_svm()
                game_map_info.total_steps = convert_steps_to_hetero(steps)
            except Exception as e:
                logging.error(e, exc_info=True)

        if isinstance(game_map_info.game_result, GameFailedDetails) or (
            len(self._games_info[map_name].total_steps) == 0 and required
        ):
            logger = logging.error
        else:
            logger = logging.debug
        self._get_and_log_proc_output(game_map_info.proc, logger)

    def delete_game_artifacts(self, game_map: GameMap):
        map_name = game_map.MapName
        dir = Path(SVMS_OUTPUT_PATH) / map_name
        _ = delete_dir(dir)

        if map_name in self._games_info:
            states = self._games_info.pop(map_name)
            del states

    def _get_proc_output(self, proc: subprocess.Popen):
        return proc.communicate()

    def _get_and_log_proc_output(
        self, proc: subprocess.Popen, logger: Callable[..., None] = logging.debug
    ):
        return self._log_proc_output(self._get_proc_output(proc), logger)

    def _log_proc_output(
        self, proc_output: tuple, logger: Callable[..., None] = logging.debug
    ):
        out, err = proc_output
        logger(f"out:\n{str(out)}\nerr:\n{str(err)}")

    def _create_preparator(self):
        return ModelGamePreparator(self._namespace, self._model, self._path_to_model)
