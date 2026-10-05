"""Unit tests for the dataset episode generation helpers."""

import pytest
from common.game import GameMap
from generate_episodes import create_episode, generate_steps, is_duplicate

pytestmark = pytest.mark.unit


def _map(steps_to_start: int = 0, strategy: str = "BFS") -> GameMap:
    return GameMap(
        StepsToPlay=1000,
        StepsToStart=steps_to_start,
        AssemblyFullName="assembly",
        NameOfObjectToCover="Method",
        DefaultSearcher=strategy,
        MapName="Method",
    )


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
