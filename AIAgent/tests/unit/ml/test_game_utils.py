"""Unit tests for the model-game step helpers.

Graph inputs are built through the shared ``gamestate_factory`` and converted by
the production ``get_hetero_data``/``update_game_state`` paths, so the tests can
not drift from the tensor schema. ``get_steps_from_svm`` reads a ``tmp_path``
file through a fake process.
"""

import pytest
from ml.validation.coverage.game_managers.model import game_utils
from ml.validation.coverage.game_managers.model.classes import (
    ModelGameMapInfo,
    ModelGameStep,
)
from ml.validation.coverage.game_managers.model.game_utils import (
    convert_steps_to_hetero,
    get_steps_from_svm,
)

pytestmark = [pytest.mark.unit, pytest.mark.serial]


def _info(proc) -> ModelGameMapInfo:
    return ModelGameMapInfo(
        total_game_state=None, total_steps=[], proc=proc, game_result=None
    )


def test_convert_steps_to_hetero_empty_returns_empty() -> None:
    assert convert_steps_to_hetero([]) == []


def test_convert_steps_to_hetero_single_step(gamestate_factory) -> None:
    step = ModelGameStep(GameState=gamestate_factory(), Output=[[0.25, 0.75]])

    hetero = convert_steps_to_hetero([step])

    assert len(hetero) == 1
    assert hetero[0]["y_true"].shape == (2, 1)


def test_convert_steps_to_hetero_updates_state_between_steps(
    gamestate_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = gamestate_factory()
    second = gamestate_factory()
    calls: list[tuple] = []

    def spy(game_state, delta):
        calls.append((game_state, delta))
        return delta

    monkeypatch.setattr(game_utils, "update_game_state", spy)
    steps = [
        ModelGameStep(GameState=first, Output=[[1.0]]),
        ModelGameStep(GameState=second, Output=[[1.0]]),
    ]

    hetero = convert_steps_to_hetero(steps)

    assert len(hetero) == 2
    assert calls == [(first, second)]


def test_get_steps_from_svm_reads_the_serialized_file(
    gamestate_factory, game_map_factory, tmp_path, fake_proc
) -> None:
    steps = [ModelGameStep(GameState=gamestate_factory(), Output=[[0.1, 0.9]])]
    (tmp_path / "MapName_steps").write_text(
        ModelGameStep.schema().dumps(steps, many=True)
    )
    proc = fake_proc()

    result = get_steps_from_svm(game_map_factory("MapName"), _info(proc), tmp_path)

    assert result == steps
    assert proc.waited is True


def test_get_steps_from_svm_returns_empty_without_a_process(
    game_map_factory, tmp_path
) -> None:
    assert get_steps_from_svm(game_map_factory("MapName"), _info(None), tmp_path) == []
