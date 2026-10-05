"""Unit tests for the each-step coverage game-state helpers."""

import pytest
from common.game import (
    GameEdgeLabel,
    GameMapEdge,
    GameMapVertex,
    GameState,
    PathConditionVertex,
    State,
)
from ml.validation.coverage.game_managers.each_step.game_states_utils import (
    get_states,
    update_game_state,
)

pytestmark = pytest.mark.unit


def _vertex(vertex_id: int, states: list[int]) -> GameMapVertex:
    return GameMapVertex(
        Id=vertex_id,
        InCoverageZone=True,
        BasicBlockSize=1,
        CoveredByTest=True,
        VisitedByState=True,
        TouchedByState=True,
        ContainsCall=False,
        ContainsThrow=False,
        States=states,
    )


def _state(state_id: int, children: list[int]) -> State:
    return State(
        Id=state_id,
        Position=state_id,
        PathCondition=[],
        VisitedAgainVertices=0,
        VisitedNotCoveredVerticesInZone=0,
        VisitedNotCoveredVerticesOutOfZone=0,
        History=[],
        Children=children,
        StepWhenMovedLastTime=0,
        InstructionsVisitedInCurrentBlock=0,
    )


def test_get_states_returns_the_state_ids(gamestate_factory):
    assert get_states(gamestate_factory()) == {0, 1}


def test_update_none_returns_the_delta(gamestate_factory):
    delta = gamestate_factory()

    assert update_game_state(None, delta) is delta


def test_update_replaces_blocks_prunes_states_and_children():
    base = GameState(
        GraphVertices=[_vertex(0, [0]), _vertex(1, [1])],
        States=[_state(0, [1]), _state(1, [0])],
        PathConditionVertices=[PathConditionVertex(Id=0, Type=0, Children=[])],
        Map=[GameMapEdge(VertexFrom=0, VertexTo=1, Label=GameEdgeLabel(Token=0))],
    )
    delta = GameState(
        GraphVertices=[_vertex(0, [1])],
        States=[_state(1, [0])],
        PathConditionVertices=[PathConditionVertex(Id=1, Type=1, Children=[])],
        Map=[GameMapEdge(VertexFrom=1, VertexTo=0, Label=GameEdgeLabel(Token=1))],
    )

    updated = update_game_state(base, delta)

    assert {vertex.Id for vertex in updated.GraphVertices} == {0, 1}
    assert next(v for v in updated.GraphVertices if v.Id == 0).States == [1]
    assert {state.Id for state in updated.States} == {1}
    assert updated.States[0].Children == []
    assert [(edge.VertexFrom, edge.VertexTo) for edge in updated.Map] == [(1, 0)]
    assert [pc.Id for pc in updated.PathConditionVertices] == [0, 1]


def test_update_keeps_states_referenced_only_by_untouched_vertices():
    base = GameState(
        GraphVertices=[_vertex(0, [0]), _vertex(1, [1])],
        States=[_state(0, []), _state(1, [])],
        PathConditionVertices=[],
        Map=[],
    )
    delta = GameState(
        GraphVertices=[_vertex(0, [0])],
        States=[_state(0, [])],
        PathConditionVertices=[],
        Map=[],
    )

    updated = update_game_state(base, delta)

    assert {state.Id for state in updated.States} == {0, 1}
