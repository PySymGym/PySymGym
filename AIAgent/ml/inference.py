class TORCH:
    game_vertex = "game_vertex"
    state_vertex = "state_vertex"
    gamevertex_to_gamevertex = (
        "game_vertex",
        "to",
        "game_vertex",
    )
    gamevertex_history_statevertex = (
        "game_vertex",
        "history",
        "state_vertex",
    )
    gamevertex_in_statevertex = (
        "game_vertex",
        "in",
        "state_vertex",
    )
    statevertex_parentof_statevertex = (
        "state_vertex",
        "parent_of",
        "state_vertex",
    )


def underscore_join(iterable):
    # compatibility with VSharp ONNX string keys
    iterable = list(map(lambda x: x.replace("_", ""), iterable))
    return "_".join(iterable)


class ONNX:
    # string values are the same as parameter names
    game_vertex = "game_vertex"
    state_vertex = "state_vertex"
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
