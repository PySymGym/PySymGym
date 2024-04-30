_GAME_VERTEX = "game_vertex"
_STATE_VERTEX = "state_vertex"


class TORCH:
    game_vertex = _GAME_VERTEX
    state_vertex = _STATE_VERTEX
    gamevertex_to_gamevertex = (_GAME_VERTEX, "to", _GAME_VERTEX)
    gamevertex_history_statevertex = (_GAME_VERTEX, "history", _STATE_VERTEX)
    gamevertex_in_statevertex = (_GAME_VERTEX, "in", _STATE_VERTEX)
    statevertex_parentof_statevertex = (_STATE_VERTEX, "parent_of", _STATE_VERTEX)
    # not used in ONNX
    statevertex_history_gamevertex = (_STATE_VERTEX, "history", _GAME_VERTEX)
    statevertex_in_gamevertex = (_STATE_VERTEX, "in", _GAME_VERTEX)


def underscore_join(iterable):
    # compatibility with VSharp ONNX string keys
    iterable = list(map(lambda x: x.replace("_", ""), iterable))
    return "_".join(iterable)


class ONNX:
    # string values are the same as parameter names
    game_vertex = TORCH.game_vertex
    state_vertex = TORCH.state_vertex
    gamevertex_to_gamevertex_index = underscore_join(
        TORCH.gamevertex_to_gamevertex + ("index",)
    )
    gamevertex_to_gamevertex_type = underscore_join(
        TORCH.gamevertex_to_gamevertex + ("type",)
    )
    gamevertex_history_statevertex_index = underscore_join(
        TORCH.gamevertex_history_statevertex + ("index",)
    )
    gamevertex_history_statevertex_attrs = underscore_join(
        TORCH.gamevertex_history_statevertex + ("attrs",)
    )
    gamevertex_in_statevertex = underscore_join(TORCH.gamevertex_in_statevertex)
    statevertex_parentof_statevertex = underscore_join(
        TORCH.statevertex_parentof_statevertex
    )


def infer(model, data):
    return model(
        game_x=data[TORCH.game_vertex].x,
        state_x=data[TORCH.state_vertex].x,
        edge_index_v_v=data[*TORCH.gamevertex_to_gamevertex].edge_index,
        edge_type_v_v=data[*TORCH.gamevertex_to_gamevertex].edge_type,
        edge_index_history_v_s=data[*TORCH.gamevertex_history_statevertex].edge_index,
        edge_attr_history_v_s=data[*TORCH.gamevertex_history_statevertex].edge_attr,
        edge_index_in_v_s=data[*TORCH.gamevertex_in_statevertex].edge_index,
        edge_index_s_s=data[*TORCH.statevertex_parentof_statevertex].edge_index,
    )
