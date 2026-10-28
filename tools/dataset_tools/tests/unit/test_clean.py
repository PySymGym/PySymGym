"""Unit tests for the dataset cleaning tool."""

import sys
from pathlib import Path

import pytest
from clean import clean, get_bad_episodes, main
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


@pytest.fixture
def run_main(monkeypatch: pytest.MonkeyPatch):
    """Run ``clean.main()`` with the given argv."""

    def _run(argv: list[str]) -> None:
        monkeypatch.setattr(sys, "argv", ["clean"] + argv)
        main()

    return _run


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


def test_main_cleans_the_dataset(run_main, tmp_path: Path) -> None:
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

    run_main(["-d", str(dataset_path), "-l", str(log)])

    result = GameMap.schema().loads(dataset_path.read_text(), many=True)
    assert [game_map.MapName for game_map in result] == ["MapC", "MapA"]


def test_main_missing_dataset_reports_dataset_error(run_main, tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing.json"
    log = tmp_path / "missing.log"

    with pytest.raises(SystemExit) as excinfo:
        run_main(["-d", str(dataset_path), "-l", str(log)])

    # The dataset check runs before the log check, so a missing dataset
    # wins even when the log is missing too.
    assert excinfo.value.args[0] == (f"Error: Dataset file not found at {dataset_path}")


def test_main_missing_log_reports_log_error(run_main, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        GameMap.schema().dumps([_map("MapA", 0)], many=True, indent=4)
    )
    log = tmp_path / "missing.log"

    with pytest.raises(SystemExit) as excinfo:
        run_main(["-d", str(dataset_path), "-l", str(log)])

    assert excinfo.value.args[0] == f"Error: Log file not found at {log}"
