"""Unit tests for the dataset episode generation helpers."""

import sys
from pathlib import Path

import pytest
from common.game import GameMap
from generate_episodes import (
    create_episode,
    generate_episodes,
    generate_steps,
    is_duplicate,
    main,
)

pytestmark = pytest.mark.unit


def _map(
    steps_to_start: int = 0,
    strategy: str = "BFS",
    method: str = "Method",
    steps_to_play: int = 1000,
    map_name: str | None = None,
) -> GameMap:
    return GameMap(
        StepsToPlay=steps_to_play,
        StepsToStart=steps_to_start,
        AssemblyFullName="assembly",
        NameOfObjectToCover=method,
        DefaultSearcher=strategy,
        MapName=map_name if map_name is not None else method,
    )


def _write_dataset(path: Path, episodes: list[GameMap]) -> None:
    path.write_text(GameMap.schema().dumps(episodes, many=True, indent=4))


def _read_dataset(path: Path) -> list[GameMap]:
    return GameMap.schema().loads(path.read_text(), many=True)


@pytest.fixture
def run_main(monkeypatch: pytest.MonkeyPatch):
    """Run ``generate_episodes.main()`` with the given argv."""

    def _run(argv: list[str]) -> None:
        monkeypatch.setattr(sys, "argv", ["generate_episodes"] + argv)
        main()

    return _run


@pytest.mark.parametrize(
    ("steps_to_play", "expected_first", "expected_step"),
    [(400, 50, 50), (500, 50, 50), (1000, 100, 100), (5000, 200, 200)],
)
def test_generate_steps_uses_bucket_size(
    steps_to_play: int, expected_first: int, expected_step: int
) -> None:
    steps = generate_steps(steps_to_play)
    assert steps[0] == expected_first
    assert steps[1] - steps[0] == expected_step
    assert steps[-1] < steps_to_play


def test_generate_steps_for_large_values_uses_thousand_step() -> None:
    steps = generate_steps(20000)
    assert steps[0] == 1000
    assert steps[1] - steps[0] == 1000
    assert steps[-1] == 19000


def test_create_episode_names_zero_step_specially() -> None:
    episode = create_episode(_map(), "Method", 0, "BFS")
    assert episode.MapName == "Method_0"
    assert episode.NameOfObjectToCover == "Method"


def test_create_episode_names_nonzero_step_with_strategy() -> None:
    episode = create_episode(_map(), "Method", 50, "DFS")
    assert episode.MapName == "Method_50_DFS"
    assert episode.StepsToStart == 50
    assert episode.DefaultSearcher == "DFS"


def test_is_duplicate_matches_method_step_and_strategy() -> None:
    existing = [_map(steps_to_start=0, strategy="BFS")]

    assert is_duplicate(existing, "Method", "BFS", 0)
    assert not is_duplicate(existing, "Method", "DFS", 0)
    assert not is_duplicate(existing, "Other", "BFS", 0)
    assert not is_duplicate(existing, "Method", "BFS", 50)


def test_generate_episodes_adds_grid_and_zero_bfs(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(
        dataset_path,
        [_map(steps_to_start=0, method="Meth", steps_to_play=100, map_name="Meth_0")],
    )

    generate_episodes(dataset_path)

    result = _read_dataset(dataset_path)
    assert [episode.MapName for episode in result] == [
        "Meth_0",
        "Meth_50_BFS",
        "Meth_50_DFS",
    ]
    by_name = {episode.MapName: episode for episode in result}
    for name, searcher in (("Meth_50_BFS", "BFS"), ("Meth_50_DFS", "DFS")):
        episode = by_name[name]
        assert episode.StepsToStart == 50
        assert episode.DefaultSearcher == searcher
        assert episode.StepsToPlay == 100
        assert episode.AssemblyFullName == "assembly"
        assert episode.NameOfObjectToCover == "Meth"


def test_generate_episodes_skips_existing_duplicates(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(
        dataset_path,
        [
            _map(
                steps_to_start=50,
                method="Meth",
                steps_to_play=100,
                map_name="Meth_50_BFS_custom",
            )
        ],
    )

    generate_episodes(dataset_path)

    result = _read_dataset(dataset_path)
    # Deduplication matches on (method, step, searcher) and ignores the
    # MapName: the custom-named (Meth, 50, BFS) episode suppresses the
    # generated Meth_50_BFS and survives in the output.
    assert [episode.MapName for episode in result] == [
        "Meth_0",
        "Meth_50_BFS_custom",
        "Meth_50_DFS",
    ]


def test_generate_episodes_empty_dataset_stays_empty(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path, [])

    generate_episodes(dataset_path)

    assert _read_dataset(dataset_path) == []


def test_generate_episodes_groups_by_method_with_own_grids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(
        dataset_path,
        [
            _map(steps_to_start=0, method="A", steps_to_play=100, map_name="A_0"),
            _map(steps_to_start=0, method="B", steps_to_play=300, map_name="B_0"),
        ],
    )

    generate_episodes(dataset_path)

    result = _read_dataset(dataset_path)
    # Each method's grid comes from its FIRST episode's StepsToPlay
    # (A: [50], B: [50, 100, 150, 200, 250]); the existing step-0 BFS
    # episodes are duplicates and not re-added.
    assert [episode.MapName for episode in result] == [
        "A_0",
        "A_50_BFS",
        "A_50_DFS",
        "B_0",
        "B_50_BFS",
        "B_50_DFS",
        "B_100_BFS",
        "B_100_DFS",
        "B_150_BFS",
        "B_150_DFS",
        "B_200_BFS",
        "B_200_DFS",
        "B_250_BFS",
        "B_250_DFS",
    ]


def test_main_rewrites_the_dataset(run_main, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(
        dataset_path,
        [_map(steps_to_start=0, method="Meth", steps_to_play=100, map_name="Meth_0")],
    )

    run_main(["-d", str(dataset_path)])

    result = _read_dataset(dataset_path)
    assert [episode.MapName for episode in result] == [
        "Meth_0",
        "Meth_50_BFS",
        "Meth_50_DFS",
    ]


def test_main_missing_dataset_prints_error_and_exits_1(
    run_main, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dataset_path = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as excinfo:
        run_main(["-d", str(dataset_path)])

    assert excinfo.value.code == 1
    assert capsys.readouterr().out == (
        f"Error: Dataset file not found at {dataset_path}\n"
    )
