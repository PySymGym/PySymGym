from common.game import GameState


def get_states(game_state: GameState) -> set[int]:
    return {s.Id for s in game_state.States}


def update_game_state(game_state: GameState, delta: GameState) -> GameState:
    if game_state is None:
        return delta

    updated_basic_blocks = {v.Id for v in delta.GraphVertices}
    updated_states = get_states(delta)

    vertices = [
        v for v in game_state.GraphVertices if v.Id not in updated_basic_blocks
    ] + delta.GraphVertices

    edges = [
        e for e in game_state.Map if e.VertexFrom not in updated_basic_blocks
    ] + delta.Map

    active_states = {state for v in vertices for state in v.States}
    new_states = [
        s
        for s in game_state.States
        if s.Id in active_states and s.Id not in updated_states
    ] + delta.States
    for s in new_states:
        s.Children = [c for c in s.Children if c in active_states]
    new_path_condition_vertices = game_state.PathConditionVertices + [
        x
        for x in delta.PathConditionVertices
        if x.Id not in list(map(lambda y: y.Id, game_state.PathConditionVertices))
    ]
    return GameState(vertices, new_states, new_path_condition_vertices, edges)
