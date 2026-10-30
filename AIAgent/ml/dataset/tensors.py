"""Domain→tensor conversion for the training dataset.

Turns a ``common.game.GameState`` (the JSON game map the servers serialize)
into the ``torch_geometric.HeteroData`` graph the models consume, plus the
small step/alias vocabulary the rest of the package shares. The feature
widths of the produced matrices come from :mod:`ml.schema`.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, TypeAlias

import numpy as np
import torch
from common.game import GameState, PathConditionVertex
from ml.inference import TORCH
from ml.schema import NUM_CFG_VERTEX_FEATURES, NUM_PC_FEATURES, NUM_STATE_FEATURES
from torch_geometric.data import HeteroData

StateId: TypeAlias = int
StateIndex: TypeAlias = int
StateMap: TypeAlias = Dict[StateId, StateIndex]

VertexId: TypeAlias = int
VertexIndex: TypeAlias = int
VertexMap: TypeAlias = Dict[VertexId, VertexIndex]

MapName: TypeAlias = str

PathConditionVertexId: TypeAlias = int
PathConditionVertexIndex: TypeAlias = int

StatesDistribution: TypeAlias = torch.tensor


@dataclass
class Step:
    graph: HeteroData
    states_distribution: StatesDistribution


def convert_input_to_tensor(
    input: GameState,
) -> Tuple[HeteroData, Dict[StateId, StateIndex]]:
    """
    Converts game env to tensors
    """
    graphVertices = input.GraphVertices
    path_condition_vertices = input.PathConditionVertices
    game_states, game_edges = input.States, input.Map
    data = HeteroData()
    nodes_vertex, edges_index_v_v, edges_types_v_v = [], [], []
    nodes_state, edges_index_s_s = [], []
    edges_index_s_v_in, edges_index_v_s_in = [], []
    edges_index_s_v_history, edges_index_v_s_history = [], []
    edges_attr_s_v = []

    nodes_path_condition = []
    edge_index_pc_pc, edge_index_pc_state, edge_index_state_pc = [], [], []

    state_map: Dict[StateId, StateIndex] = dict()
    vertex_map: Dict[VertexId, VertexIndex] = dict()
    vertex_index = 0
    state_index = 0

    def one_hot_encoding(pc_v: PathConditionVertex) -> np.ndarray:
        encoded = np.zeros(NUM_PC_FEATURES, dtype=int)
        encoded[pc_v.Type] = 1
        return encoded

    pc_map: Dict[PathConditionVertexId, PathConditionVertexIndex] = dict()
    for pc_idx, pc_v in enumerate(path_condition_vertices):
        if pc_v.Id not in pc_map:
            nodes_path_condition.append(one_hot_encoding(pc_v))
            pc_map[pc_v.Id] = pc_idx
        else:
            raise RuntimeError(
                "There are two path condition vertices with the same Id."
            )

    for pc_v in path_condition_vertices:
        for child_id in pc_v.Children:
            edge_index_pc_pc.extend(
                [
                    [pc_map[pc_v.Id], pc_map[child_id]],
                    [pc_map[child_id], pc_map[pc_v.Id]],
                ]
            )

    # vertex nodes
    for v in graphVertices:
        vertex_id = v.Id
        if vertex_id not in vertex_map:
            vertex_map[vertex_id] = vertex_index  # maintain order in tensors
            vertex_index = vertex_index + 1
            nodes_vertex.append(
                np.array(
                    [
                        int(v.InCoverageZone),
                        v.BasicBlockSize,
                        int(v.CoveredByTest),
                        int(v.VisitedByState),
                        int(v.TouchedByState),
                        int(v.ContainsCall),
                        int(v.ContainsThrow),
                    ]
                )
            )
    # vertex -> vertex edges
    for e in game_edges:
        edges_index_v_v.append(
            np.array([vertex_map[e.VertexFrom], vertex_map[e.VertexTo]])
        )
        edges_types_v_v.append(e.Label.Token)

    state_doubles = 0

    # state nodes
    for s in game_states:
        state_id = s.Id
        if state_id not in state_map:
            state_map[state_id] = state_index
            nodes_state.append(
                np.array(
                    [
                        s.Position,
                        s.VisitedAgainVertices,
                        s.VisitedNotCoveredVerticesInZone,
                        s.VisitedNotCoveredVerticesOutOfZone,
                        s.StepWhenMovedLastTime,
                        s.InstructionsVisitedInCurrentBlock,
                    ]
                )
            )

            # history edges: state -> vertex and back
            for h in s.History:
                v_to = vertex_map[h.GraphVertexId]
                edges_index_s_v_history.append(np.array([state_index, v_to]))
                edges_index_v_s_history.append(np.array([v_to, state_index]))
                edges_attr_s_v.append(
                    np.array([h.NumOfVisits, h.StepWhenVisitedLastTime])
                )
            state_index = state_index + 1

            # pc edges: state -> pc vertex and back
            for pc_id in s.PathCondition:
                edge_index_pc_state.append([pc_map[pc_id], state_map[state_id]])
                edge_index_state_pc.append([state_map[state_id], pc_map[pc_id]])

        else:
            state_doubles += 1

    # state and its childen edges: state -> state
    for s in game_states:
        for ch in s.Children:
            # A child Id absent from state_map is a malformed game state;
            # the KeyError (keyed by the missing child Id) is the signal.
            edges_index_s_s.append(np.array([state_map[s.Id], state_map[ch]]))

    # state position edges: vertex -> state and back
    for v in graphVertices:
        for s in v.States:
            edges_index_s_v_in.append(np.array([state_map[s], vertex_map[v.Id]]))
            edges_index_v_s_in.append(np.array([vertex_map[v.Id], state_map[s]]))

    data[TORCH.game_vertex].x = torch.tensor(
        np.array(nodes_vertex, dtype=float).reshape(-1, NUM_CFG_VERTEX_FEATURES),
        dtype=torch.float,
    )
    data[TORCH.state_vertex].x = torch.tensor(
        np.array(nodes_state, dtype=float).reshape(-1, NUM_STATE_FEATURES),
        dtype=torch.float,
    )
    data[TORCH.path_condition_vertex].x = torch.tensor(
        np.array(nodes_path_condition, dtype=float).reshape(-1, NUM_PC_FEATURES),
        dtype=torch.float,
    )

    # dumb fix
    def null_if_empty(tensor):
        return tensor if tensor.numel() != 0 else torch.empty((2, 0), dtype=torch.int64)

    data[*TORCH.gamevertex_to_gamevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_v_v), dtype=torch.long).t().contiguous()
    )
    data[*TORCH.gamevertex_to_gamevertex].edge_type = torch.tensor(
        np.array(edges_types_v_v), dtype=torch.long
    )
    data[*TORCH.statevertex_in_gamevertex].edge_index = (
        torch.tensor(np.array(edges_index_s_v_in), dtype=torch.long).t().contiguous()
    )
    data[*TORCH.gamevertex_in_statevertex].edge_index = (
        torch.tensor(np.array(edges_index_v_s_in), dtype=torch.long).t().contiguous()
    )
    data[*TORCH.statevertex_history_gamevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_s_v_history), dtype=torch.long)
        .t()
        .contiguous()
    )
    data[*TORCH.gamevertex_history_statevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_v_s_history), dtype=torch.long)
        .t()
        .contiguous()
    )
    data[*TORCH.statevertex_history_gamevertex].edge_attr = torch.tensor(
        np.array(edges_attr_s_v), dtype=torch.long
    )
    data[*TORCH.gamevertex_history_statevertex].edge_attr = torch.tensor(
        np.array(edges_attr_s_v), dtype=torch.long
    )
    # if (edges_index_s_s): #TODO: empty?
    data[*TORCH.statevertex_parentof_statevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_s_s), dtype=torch.long).t().contiguous()
    )
    data[TORCH.pathcondvertex_to_pathcondvertex].edge_index = null_if_empty(
        torch.tensor(np.array(edge_index_pc_pc), dtype=torch.long).t().contiguous()
    )
    data[TORCH.pathcondvertex_to_statevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edge_index_pc_state), dtype=torch.long).t().contiguous()
    )
    data[TORCH.statevertex_to_pathcondvertex].edge_index = null_if_empty(
        torch.tensor(np.array(edge_index_state_pc), dtype=torch.long).t().contiguous()
    )
    return data, state_map


def remove_extra_attrs(step: HeteroData):
    if hasattr(step[TORCH.statevertex_history_gamevertex], "edge_attr"):
        del step[TORCH.statevertex_history_gamevertex].edge_attr
    if hasattr(step[TORCH.gamevertex_to_gamevertex], "edge_attr"):
        del step[TORCH.gamevertex_to_gamevertex].edge_attr
    if hasattr(step, "use_for_train"):
        del step.use_for_train


def get_hetero_data(game_state: GameState, nn_output: list[list[float]]) -> HeteroData:
    """
    Attach the model output to the converted graph as the ``y_true`` column.

    Parameters
    ----------
    game_state : GameState
        The game state to convert.
    nn_output : list[list[float]]
        The model's single output tensor flattened to one row of per-state
        values (the shape the V# producer,
        ``AISearcher.convertOutputToJson``, writes). Transposed to the
        ``[S, 1]`` ``y_true`` column.

    Raises
    ------
    ValueError
        If ``nn_output`` does not have exactly one row.
    """
    if len(nn_output) != 1:
        raise ValueError(
            "nn_output must have exactly one row of per-state values, "
            f"got {len(nn_output)}"
        )
    hetero_input, _ = convert_input_to_tensor(game_state)
    nn_output = [[x] for x in nn_output[0]]
    hetero_input["y_true"] = torch.Tensor(nn_output)
    return hetero_input
