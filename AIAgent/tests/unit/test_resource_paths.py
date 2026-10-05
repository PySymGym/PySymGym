"""Resource paths must be absolute and injectable, never CWD-relative.

The ONNX reference gamestate used to be located through a path relative to the
current working directory, so it broke whenever a test or tool ran from
elsewhere. These tests pin the path to the repository and prove the filesystem
seams can be redirected through the constructors.
"""

import threading
from types import SimpleNamespace

import pytest

import paths
from common.game import GameMap
from ml.validation.coverage.game_managers.model import process_game_manager as pgm

pytestmark = pytest.mark.unit


def _namespace() -> SimpleNamespace:
    return SimpleNamespace(
        shared_lock=threading.Lock(),
        is_prepared=SimpleNamespace(value=False),
    )


def test_resources_root_is_absolute_and_exists() -> None:
    assert paths.RESOURCES_PATH.is_absolute()
    assert paths.RESOURCES_PATH.is_dir()


def test_gamestate_example_is_absolute_and_under_resources() -> None:
    assert pgm.GAMESTATE_EXAMPLE_PATH.is_absolute()
    assert pgm.GAMESTATE_EXAMPLE_PATH.is_file()
    assert paths.RESOURCES_PATH in pgm.GAMESTATE_EXAMPLE_PATH.parents


def test_paths_survive_a_cwd_change(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert paths.RESOURCES_PATH.is_dir()
    assert pgm.GAMESTATE_EXAMPLE_PATH.is_file()


def test_preparator_and_manager_paths_are_injectable(tmp_path) -> None:
    gamestate = tmp_path / "example.json"
    svms_output = tmp_path / "svms"
    namespace = _namespace()

    preparator = pgm.ModelGamePreparator(
        namespace,
        model=None,
        gamestate_example_path=gamestate,
        svms_output_path=svms_output,
    )
    assert preparator._gamestate_example_path == gamestate
    assert preparator._svms_output_path == svms_output

    manager = pgm.ModelGameManager(namespace, model=None, svms_output_path=svms_output)
    assert manager._preparator._svms_output_path == svms_output
    game_map = GameMap(
        StepsToPlay=1,
        StepsToStart=0,
        AssemblyFullName="a",
        NameOfObjectToCover="o",
        DefaultSearcher="BFS",
        MapName="m",
    )
    assert manager._get_output_dir(game_map) == svms_output / "m"
