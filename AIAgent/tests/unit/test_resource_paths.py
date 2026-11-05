"""Resource paths must be absolute and injectable, never CWD-relative.

The ONNX reference gamestate used to be located through a path relative to the
current working directory, so it broke whenever a test or tool ran from
elsewhere. These tests pin the path to the repository and prove the filesystem
seams can be redirected through the constructors.
"""

import threading
from multiprocessing.managers import Namespace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import paths
from common.config.feature_config import FeatureConfig
from common.game import GameMap
from ml.validation.coverage.game_managers.model import process_game_manager as pgm

pytestmark = pytest.mark.unit


def _namespace() -> Namespace:
    return Namespace(
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

    # The model is never touched by these path tests: an opaque sentinel
    # stands in for the required ``torch.nn.Module``.
    model = cast(torch.nn.Module, object())
    preparator = pgm.ModelGamePreparator(
        namespace,
        model=model,
        gamestate_example_path=gamestate,
        svms_output_path=svms_output,
    )
    assert preparator._gamestate_example_path == gamestate
    assert preparator._svms_output_path == svms_output

    manager = pgm.ModelGameManager(
        namespace,
        model=model,
        feature_config=FeatureConfig(),
        svms_output_path=svms_output,
    )
    # ``_create_preparator`` returns the concrete preparator; the base
    # class only types it as ``BaseGamePreparator``.
    manager_preparator = cast(pgm.ModelGamePreparator, manager._preparator)
    assert manager_preparator._svms_output_path == svms_output
    game_map = GameMap(
        StepsToPlay=1,
        StepsToStart=0,
        AssemblyFullName="a",
        NameOfObjectToCover="o",
        DefaultSearcher="BFS",
        MapName="m",
    )
    assert manager._get_output_dir(game_map) == svms_output / "m"
