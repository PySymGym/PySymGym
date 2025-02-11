import json
import logging
import socket
import subprocess
from dataclasses import dataclass
from multiprocessing.managers import Namespace
from pathlib import Path
from typing import Callable, Optional

import torch
import yaml
from common.classes import GameFailed, GameResult, Map2Result
from common.file_system_utils import delete_dir
from common.game import GameMap, GameMap2SVM, GameState
from common.network_utils import next_free_port
from func_timeout import FunctionTimedOut
from ml.dataset import convert_input_to_tensor
from ml.validation.coverage.game_managers.base_game_manager import (
    BaseGameManager,
    BaseGamePreparator,
)
from ml.validation.coverage.game_managers.each_step.game_states_utils import (
    update_game_state,
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
        path_to_model: str = CURRENT_MODEL_PATH,
    ):
        self._model = model
        self._path_to_model = path_to_model
        super().__init__(namespace)

    def _create_onnx_model(self):
        import_model_fqn = self._model.__class__.__module__ + ".StateModelEncoder"
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


@dataclass
class ModelGameMapInfo:
    total_game_state: GameState
    total_steps: list[HeteroData]
    occupied_port: int


class ModelGameManager(BaseGameManager):
    def __init__(
        self,
        namespace: Namespace,
        model: torch.nn.Module,
        path_to_model: str = CURRENT_MODEL_PATH,
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
        try:
            # TODO: rewrite it more clear to read
            running_res = self._run_game_process(game_map2svm)
            proc, port, socket = running_res
            self._games_info[map_name] = ModelGameMapInfo(
                total_game_state=None, total_steps=[], occupied_port=port
            )
            self._wait_for_game_over(proc, game_map, socket)
            game_result = self._get_result(game_map2svm)
            if (
                isinstance(game_result, GameFailed)
                or len(self._games_info[map_name].total_steps) == 0
            ):
                logger = logging.error
            else:
                logger = logging.debug
            self._get_and_log_proc_output(proc, logger)
        except FunctionTimedOut as error:
            logging.warning(f"Timeouted on map {map_name} with {error.timedOutAfter}s")
            self._kill_game_process(proc, logging.warning)
            game_result = GameFailed(reason=type(error))
        except Exception as e:
            logging.error(e, exc_info=True)
            if running_res is not None:
                self._kill_game_process(proc, logging.error)
            game_result = GameFailed(reason=type(e))
        torch.cuda.empty_cache()
        return Map2Result(game_map2svm, game_result)

    def _get_output_dir(self, game_map: GameMap) -> Path:
        return SVMS_OUTPUT_PATH / f"{game_map.MapName}"

    def _run_game_process(self, game_map2svm: GameMap2SVM):
        game_map, svm_info = game_map2svm.GameMap, game_map2svm.SVMInfo
        svm_info = game_map2svm.SVMInfo

        def look_for_free_port(attempts=100) -> int:
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
                return look_for_free_port(attempts - 1)

        port, server_socket = look_for_free_port()
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
    def _wait_for_game_over(
        self, proc: subprocess.Popen, game_map: GameMap, socket: socket.socket
    ):
        self._get_steps_while_playing(game_map, socket)
        proc.wait()
        assert proc.returncode == 0

    def _get_steps_while_playing(self, game_map: GameMap, server_socket: socket.socket):
        map_name = game_map.MapName
        game_map_info = self._games_info[map_name]
        game_state, steps, port = (
            game_map_info.total_game_state,
            game_map_info.total_steps,
            game_map_info.occupied_port,
        )
        step_count = len(steps)

        logging.debug(f"Server listening on port {port}")
        conn, addr = server_socket.accept()
        logging.debug(f"Connection from {addr}")

        terminator = b"\n"
        data = b""

        def get_next_message() -> Optional[bytes]:
            nonlocal data
            while terminator not in data:
                chunk = conn.recv(2048)
                if not chunk:
                    return None
                data += chunk
            message, _, data = data.partition(terminator)
            return message

        def get_game_state() -> Optional[GameState]:
            message = get_next_message()
            if message is not None:
                return GameState.from_dict(json.loads(message.decode("utf-8")))

        def get_nn_output() -> Optional[list]:
            message = get_next_message()
            if message is not None:
                return json.loads(message.decode("utf-8"))

        while step_count <= game_map.StepsToPlay:
            try:
                received_game_state = get_game_state()
                if received_game_state is None:
                    break
                logging.debug(f"<- state: {received_game_state}")
                nn_output = get_nn_output()
                logging.debug(f"<- nn_output: {nn_output}")
            except ConnectionResetError as e:
                logging.error(e, exc_info=True)
                raise
            if step_count == 0:
                game_state = received_game_state
            else:
                delta = received_game_state
                game_state = update_game_state(game_state, delta)
            hetero_input, _ = convert_input_to_tensor(game_state)
            nn_output = [[x] for x in nn_output[0]]
            hetero_input["y_true"] = torch.Tensor(nn_output)
            steps.append(hetero_input)
            step_count += 1
        self._games_info[map_name].total_game_state = game_state

    def _get_result(self, game_map2svm: GameMap2SVM) -> GameResult | GameFailed:
        game_map = game_map2svm.GameMap
        map_name = game_map.MapName
        output_dir = self._get_output_dir(game_map)
        result_file = (
            output_dir / f"{map_name}result"
        )  # TODO: the path to result must be documented
        try:
            with open(result_file) as f:
                actual_coverage_percent, tests_count, _, errors_count = f.read().split()
                actual_coverage_percent = int(actual_coverage_percent)
                tests_count = int(tests_count)
                steps_count = len(self._games_info[map_name].total_steps)
                errors_count = int(errors_count)
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
                            f"There is neither any step chosen by oracle, nor full coverage. Actual coverage: {actual_coverage_formatted}. Immediate GameOver on {map_name}."
                        )
                        return GameResult(game_map2svm.GameMap.StepsToPlay, 0, 0, 0)
                    else:
                        logging.warning(
                            f"The map {map_name} was completely covered without an oracle?!"
                        )
                elif steps_count < game_map.StepsToPlay:
                    logging.warning(
                        f"Not all steps exhausted on {game_map.MapName} with non-100% coverage. Steps taken by oracle: {steps_count}, actual coverage: {actual_coverage_formatted}."
                    )
                logging.info(
                    f"Finished map {map_name} "
                    f"in {result.steps_count} steps,"
                    f"actual coverage: {result.actual_coverage_percent:.2f}"
                )
        except FileNotFoundError:
            logging.error(
                f"The result of {map_name} cannot be found after completing the game?!"
            )
            result = GameFailed("There is no file with game result")
        return result

    def get_game_steps(self, game_map: GameMap) -> Optional[list[HeteroData]]:
        map_name = game_map.MapName
        if map_name in self._games_info:
            game_map_info = self._games_info[map_name]
            steps = game_map_info.total_steps if game_map_info.total_steps else None
        else:
            steps = None
        return steps

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
