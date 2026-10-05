"""Unit tests for the game data models in ``common.game``."""

import pytest

from common.game import MoveReward, Reward, State, StateHistoryElem

pytestmark = pytest.mark.unit


def _state(state_id: int = 7) -> State:
    return State(
        Id=state_id,
        Position=0,
        PathCondition=[1, 2],
        VisitedAgainVertices=0,
        VisitedNotCoveredVerticesInZone=0,
        VisitedNotCoveredVerticesOutOfZone=0,
        History=[
            StateHistoryElem(GraphVertexId=1, NumOfVisits=2, StepWhenVisitedLastTime=3)
        ],
        Children=[2],
        StepWhenMovedLastTime=0,
        InstructionsVisitedInCurrentBlock=0,
    )


def test_state_hash_is_defined_by_id() -> None:
    assert hash(_state(7)) == hash(7)


def test_states_with_equal_fields_collapse_in_a_set() -> None:
    assert len({_state(7), _state(7)}) == 1


def test_state_json_round_trip() -> None:
    state = _state()
    assert State.from_json(state.to_json()) == state


def test_reward_json_round_trip() -> None:
    reward = Reward(
        ForMove=MoveReward(ForCoverage=1, ForVisitedInstructions=5),
        MaxPossibleReward=9,
    )
    assert Reward.from_json(reward.to_json()) == reward


@pytest.mark.parametrize(
    ("verbose", "expected"),
    [(False, "#vi=5"), (True, "ForVisitedInstructions: 5")],
)
def test_move_reward_printable(verbose: bool, expected: str) -> None:
    reward = MoveReward(ForCoverage=3, ForVisitedInstructions=5)
    assert reward.printable(verbose) == expected
