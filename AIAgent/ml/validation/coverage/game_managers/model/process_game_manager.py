import json
import logging
import socket
import subprocess
import time
from multiprocessing.managers import Namespace
from pathlib import Path
from typing import Callable, Optional

import torch
import yaml
from common.classes import GameFailed, GameResult, Map2Result
from common.file_system_utils import delete_dir
from common.game import GameMap, GameMap2SVM
from common.network_utils import look_for_free_port_locked
from func_timeout import FunctionTimedOut
from ml.dataset import get_hetero_data
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
from onyx import (
    entrypoint as save_torch_model_to_onnx_file,
)
from onyx import (
    load_gamestate,
    resolve_import_model,
)
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
            save_torch_model_to_onnx_file(
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
        self._clean_output_folder()
        self._create_onnx_model()


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
        self._games_info: dict[str, ModelGameMapInfo] = dict()
        super().__init__(self._namespace)

    def _play_game_map(self, game_map2svm: GameMap2SVM) -> Map2Result:
        game_map = game_map2svm.GameMap
        map_name = game_map.MapName
        game_result = None
        game_info = ModelGameMapInfo(
            total_game_state=None, total_steps=[], proc=None, game_result=game_result
        )
        self._games_info[map_name] = game_info

        def kill_proc(proc: Optional[subprocess.Popen], logger: Callable[..., None]):
            if proc is None:
                logging.warning(
                    f"There is no such proc?! Can't kill instance of {map_name}."
                )
                return
            self._kill_game_process(proc, logger)

        try:
            proc, port, socket = self._run_game_process(game_map2svm)
        except Exception as e:
            logging.error(e, exc_info=True)
            return Map2Result(
                game_map2svm,
                GameFailed(reason=f"Failed to instantiate process for {map_name}"),
            )

        try:
            game_info.proc = proc
            svm_connection_info = SVMConnectionInfo(occupied_port=port, socket=socket)
            game_success_or_failure = self._get_result(game_map2svm, proc)
            # TODO: There is confusion here because of the ambiguous name of the 'GameResult'. Renaming is needed.
            if isinstance(game_success_or_failure, GameResult):
                game_result = GameResultDetails(
                    game_result=game_success_or_failure,
                    svm_connection_info=svm_connection_info,
                )
            else:
                game_result = GameFailedDetails(game_failed=game_success_or_failure)
        except FunctionTimedOut as error:
            logging.warning(
                f"Timeouted on map {map_name} with {error.timedOutAfter}s after {len(game_info.total_steps)} steps"
            )
            kill_proc(proc, logging.warning)
            game_result = GameFailedDetails(game_failed=GameFailed(reason=type(error)))
        except Exception as error:
            logging.error(error, exc_info=True)
            kill_proc(proc, logging.error)
            game_result = GameFailedDetails(game_failed=GameFailed(reason=type(error)))
        finally:
            torch.cuda.empty_cache()
        game_info.game_result = game_result

        if isinstance(game_result, GameResultDetails):
            game_result = game_result.game_result
        else:
            game_result = game_result.game_failed
        return Map2Result(game_map2svm, game_result)

    def _get_output_dir(self, game_map: GameMap) -> Path:
        return SVMS_OUTPUT_PATH / f"{game_map.MapName}"

    def _run_game_process(self, game_map2svm: GameMap2SVM):
        game_map, svm_info = game_map2svm.GameMap, game_map2svm.SVMInfo

        port, server_socket = look_for_free_port_locked(self._shared_lock, svm_info)
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

    def notify_steps_requirement(self, game_map: GameMap, required: bool):
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

            def get_steps_from_svm() -> list[ModelGameStep]:
                proc = game_map_info.proc
                if proc is None:
                    logging.error(
                        "I can't get the steps of a failed game (there was no game)"
                    )
                    return []
                output_dir = self._get_output_dir(game_map)
                steps = output_dir / f"{map_name}_steps"
                proc.wait()
                with open(steps, "br") as f:
                    steps_json = json.load(f)
                steps = list(map(lambda v: ModelGameStep.from_dict(v), steps_json))  # type: ignore
                return steps

            def convert_steps_to_hetero(steps: list[ModelGameStep]) -> list[HeteroData]:
                if not steps:
                    return []
                step = steps.pop(0)
                game_state, nn_output = step.GameState, step.Output

                steps_hetero = [get_hetero_data(game_state, nn_output)]
                for step in steps:
                    delta = step.GameState
                    game_state = update_game_state(game_state, delta)
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
        self,
        proc: Optional[subprocess.Popen],
        logger: Callable[..., None] = logging.debug,
    ):
        if proc is None:
            logger("There is no proc?! Can't log proc output.")
            return
        return self._log_proc_output(self._get_proc_output(proc), logger)

    def _log_proc_output(
        self, proc_output: tuple, logger: Callable[..., None] = logging.debug
    ):
        out, err = proc_output
        logger(f"out:\n{str(out)}\nerr:\n{str(err)}")

    def _create_preparator(self):
        return ModelGamePreparator(self._namespace, self._model, self._path_to_model)
