import json
import logging
import os
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass
from multiprocessing.managers import Namespace
from pathlib import Path
from typing import Optional

import torch
from common.classes import GameFailed, GameResult, Map2Result
from common.game import GameMap, GameMap2SVM, GameState
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


@dataclass
class ModelGameMapInfo:
    game_state: GameState
    steps: list[HeteroData]
    port: int


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
        self._occupied_ports: list[int] = namespace.occupied_ports
        self._shared_lock = namespace.shared_lock
        self._model = model
        self._path_to_model = path_to_model
        self._game_steps: dict[str, ModelGameMapInfo] = {}
        super().__init__(self._namespace)

    def _get_output_dir(self, game_map: GameMap) -> Path:
        return SVMS_OUTPUT_PATH / f"{game_map.MapName}"

    def _run_game_process(self, game_map2svm: GameMap2SVM):
        game_map, svm_info = game_map2svm.GameMap, game_map2svm.SVMInfo
        svm_info = game_map2svm.SVMInfo

        def look_for_free_port(attempts_left=100) -> int:
            if attempts_left == 0:
                raise RuntimeError("Can't find any free port!")
            logging.debug(f"Looking for port... attempts left: {attempts_left}")
            for i in range(svm_info.min_port, svm_info.max_port + 1):
                with self._shared_lock:
                    if i not in self._occupied_ports:
                        self._occupied_ports.append(i)
                        return i
            else:
                time.sleep(0.1)
                return look_for_free_port(attempts_left - 1)

        port = look_for_free_port()
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
            cwd=Path(svm_info.server_working_dir),
            encoding="utf-8",
        )
        return proc, port

    def _kill_game_process(self, proc: subprocess.Popen):
        proc.kill()
        _ = self.log_proc_output(proc)
        logging.warning(f"Process {proc.pid} was intentionally killed")

    def log_proc_output(self, proc: subprocess.Popen):
        out, err = proc.communicate()
        logging.debug(f"out = {str(out)}\nerr = {str(err)}")
        return out, err

    @set_timeout_if_needed
    def _wait_for_game_over(self, proc: subprocess.Popen, game_map: GameMap) -> None:
        self._wait_for_game_over_and_read_game_steps(proc, game_map)
        out, err = self.log_proc_output(proc)
        return out, err

    def _wait_for_game_over_and_read_game_steps(
        self, proc: subprocess.Popen, game_map: GameMap
    ):
        logging.debug("reading of steps...")
        start = time.time()
        self._get_steps_while_playing(game_map)
        end = time.time()
        steps_reading = end - start
        logging.debug(f"steps are read... {steps_reading=}s.")
        proc.wait()
        assert proc.returncode == 0

    def _get_steps_while_playing(self, game_map: GameMap):
        game_map_info = self._game_steps[game_map.MapName]
        game_state, steps, port = (
            game_map_info.game_state,
            game_map_info.steps,
            game_map_info.port,
        )
        step_number = len(steps)

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("localhost", port))
        server_socket.listen(1)
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

        while True:
            readed_game_state = get_game_state()
            if readed_game_state is None:
                break
            nn_output = get_nn_output()
            if step_number == 0:
                game_state = readed_game_state
            else:
                delta = readed_game_state
                game_state = update_game_state(game_state, delta)
            hetero_input, _ = convert_input_to_tensor(game_state)
            nn_output = list(map(lambda x: [x], nn_output[0]))
            hetero_input["y_true"] = torch.Tensor(nn_output)
            steps.append(hetero_input)
            step_number += 1
        self._game_steps[game_map.MapName].game_state = game_state

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
        proc, port = self._run_game_process(game_map2svm)
        try:
            game_map = game_map2svm.GameMap
            self._game_steps[game_map.MapName] = ModelGameMapInfo(None, [], port)
            out, err = self._wait_for_game_over(proc, game_map)
            game_result = self._get_result(game_map2svm)
            if isinstance(game_result, GameFailed):
                logging.error(f"\nout = {str(out)}\nerr = {str(err)}")
        except FunctionTimedOut as error:
            logging.warning(
                f"Timeouted on map {game_map.MapName} with {error.timedOutAfter}s"
            )
            self._kill_game_process(proc)
            game_result = GameFailed(reason=type(error))
        except Exception as e:
            trace = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
            logging.error(trace)
            self._kill_game_process(proc)
            game_result = GameFailed(reason=type(e))
        finally:
            with self._shared_lock:
                self._occupied_ports.remove(port)

        return Map2Result(game_map2svm, game_result)

    def get_game_steps(self, game_map: GameMap) -> list[HeteroData]:
        game_map_info = self._game_steps[game_map.MapName]
        return game_map_info.steps

    def _create_preparator(self):
        return ModelGamePreparator(self._namespace, self._model, self._path_to_model)

    def delete_game_steps(self, game_map: GameMap):
        directory = Path(SVMS_OUTPUT_PATH) / game_map.MapName
        empty_dir = Path(SVMS_OUTPUT_PATH) / "empty_temp_dir_for_rsync"
        os.makedirs(empty_dir, exist_ok=True)
        try:
            _ = subprocess.run(
                ["rsync", "-a", "--delete", f"{empty_dir}/", f"{directory}/"],
                check=True,
                capture_output=True,
            )
            # logging.debug(f"Directory deleted: {directory}")
        except subprocess.CalledProcessError as e:
            logging.error(
                "Something happened during deletion:"
                + "\n".join(traceback.format_exception(e.__class__, e, e.__traceback__))
            )

        states = self._game_steps.pop(game_map.MapName)
        del states
