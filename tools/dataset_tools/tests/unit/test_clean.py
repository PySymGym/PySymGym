"""Unit tests for the dataset cleaning tool."""

from pathlib import Path

import pytest
from clean import clean, get_bad_episodes
from common.game import GameMap

pytestmark = pytest.mark.unit


def _map(map_name: str, steps_to_start: int) -> GameMap:
    return GameMap(
        StepsToPlay=100,
        StepsToStart=steps_to_start,
        AssemblyFullName="assembly",
        NameOfObjectToCover="Method",
        DefaultSearcher="BFS",
        MapName=map_name,
    )


def test_get_bad_episodes_returns_empty_for_missing_log(tmp_path: Path) -> None:
    assert get_bad_episodes(tmp_path / "missing.log") == []


def test_get_bad_episodes_extracts_and_deduplicates_names(tmp_path: Path) -> None:
    log = tmp_path / "app.log"
    log.write_text(
        "immediate GameOver on MapA at step 1\n"
        "other immediate GameOver on MapA again\n"
        "unrelated line\n"
        "immediate GameOver on MapB x\n"
    )

    assert get_bad_episodes(log) == ["MapA", "MapB"]


def test_clean_removes_bad_episodes_and_sorts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        GameMap.schema().dumps(
            [_map("MapA", 20), _map("MapB", 0), _map("MapC", 10)],
            many=True,
            indent=4,
        )
    )
    log = tmp_path / "app.log"
    log.write_text("immediate GameOver on MapB\n")

    clean(log, dataset_path)

    result = GameMap.schema().loads(dataset_path.read_text(), many=True)
    assert [game_map.MapName for game_map in result] == ["MapC", "MapA"]
