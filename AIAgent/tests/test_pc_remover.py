import json
import pytest
from pathlib import Path
from ml.pc_remover import remove_path_condition_root 
import torch
# from ml.dataset import convert_input_to_tensor
# from common.game import GameState
# from ml.pc_remover import remove_path_condition_root
# from torch_geometric.data import HeteroData

# def func(x):
#     return x + 1

# def test_answer():
#     assert func(3) == 4

from dataclasses import dataclass

#from common.validation_coverage.svm_info import SVMInfo
from dataclasses_json import dataclass_json

import json
from dataclasses import dataclass 
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)


import torch


@dataclass_json
@dataclass(slots=True)
class StateHistoryElem:
    GraphVertexId: int
    NumOfVisits: int
    StepWhenVisitedLastTime: int


@dataclass_json
@dataclass(slots=True)
class PathConditionVertex:
    Id: int
    Type: int
    Children: list[int]


@dataclass_json
@dataclass(slots=True)
class State:
    Id: int
    Position: int
    PathCondition: PathConditionVertex
    VisitedAgainVertices: int
    VisitedNotCoveredVerticesInZone: int
    VisitedNotCoveredVerticesOutOfZone: int
    History: list[StateHistoryElem]
    Children: list[int]
    StepWhenMovedLastTime: int
    InstructionsVisitedInCurrentBlock: int

    def __hash__(self) -> int:
        return self.Id.__hash__()


@dataclass_json
@dataclass(slots=True)
class GameMapVertex:
    Id: int
    InCoverageZone: bool
    BasicBlockSize: int
    CoveredByTest: bool
    VisitedByState: bool
    TouchedByState: bool
    ContainsCall: bool
    ContainsThrow: bool
    States: list[int]


@dataclass_json
@dataclass(slots=True)
class GameEdgeLabel:
    Token: int


@dataclass_json
@dataclass(slots=True)
class GameMapEdge:
    VertexFrom: int
    VertexTo: int
    Label: GameEdgeLabel


@dataclass_json
@dataclass(slots=True)
class GameState:
    GraphVertices: list[GameMapVertex]
    States: list[State]
    PathConditionVertices: list[PathConditionVertex]
    Map: list[GameMapEdge]


@dataclass_json
@dataclass(slots=True)
class GameMap:
    StepsToPlay: int
    StepsToStart: int
    AssemblyFullName: str
    NameOfObjectToCover: str
    DefaultSearcher: str
    MapName: str



import torch_geometric
from torch_geometric.data import Dataset, HeteroData

GAMESTATESUFFIX = "_gameState"
STATESUFFIX = "_statesInfo"
MOVEDSTATESUFFIX = "_movedState"

NUM_PC_FEATURES = 48
NUM_STATE_FEATURES = 6
NUM_CFG_VERTEX_FEATURES = 7

_GAME_VERTEX = "game_vertex"
_STATE_VERTEX = "state_vertex"
_PATH_CONDITION_VERTEX = "path_condition_vertex"


class TORCH:
    game_vertex = _GAME_VERTEX
    state_vertex = _STATE_VERTEX
    path_condition_vertex = _PATH_CONDITION_VERTEX
    gamevertex_to_gamevertex = (_GAME_VERTEX, "to", _GAME_VERTEX)
    gamevertex_history_statevertex = (_GAME_VERTEX, "history", _STATE_VERTEX)
    gamevertex_in_statevertex = (_GAME_VERTEX, "in", _STATE_VERTEX)
    statevertex_parentof_statevertex = (_STATE_VERTEX, "parent_of", _STATE_VERTEX)
    pathcondvertex_to_pathcondvertex = (
        _PATH_CONDITION_VERTEX,
        "to",
        _PATH_CONDITION_VERTEX,
    )
    pathcondvertex_to_statevertex = (_PATH_CONDITION_VERTEX, "to", _STATE_VERTEX)
    statevertex_to_pathcondvertex = (_STATE_VERTEX, "to", _PATH_CONDITION_VERTEX)
    # not used in ONNX
    statevertex_history_gamevertex = (_STATE_VERTEX, "history", _GAME_VERTEX)
    statevertex_in_gamevertex = (_STATE_VERTEX, "in", _GAME_VERTEX)


StateId: TypeAlias = int
StateIndex: TypeAlias = int
StateMap: TypeAlias = Dict[StateId, StateIndex]

VertexId: TypeAlias = int
VertexIndex: TypeAlias = int
VertexMap: TypeAlias = Dict[VertexId, VertexIndex]

PathConditionVertexId: TypeAlias = int
PathConditionVertexIndex: TypeAlias = int

MapName: TypeAlias = str


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

            edge_index_pc_state.append(
                [pc_map[s.PathCondition.Id], state_map[state_id]]
            )
            edge_index_state_pc.append(
                [state_map[state_id], pc_map[s.PathCondition.Id]]
            )
        else:
            state_doubles += 1

    # state and its childen edges: state -> state
    for s in game_states:
        for ch in s.Children:
            try:
                edges_index_s_s.append(np.array([state_map[s.Id], state_map[ch]]))
            except KeyError:
                print("[ERROR]: KeyError")
                return None, None

    # state position edges: vertex -> state and back
    for v in graphVertices:
        for s in v.States:
            edges_index_s_v_in.append(np.array([state_map[s], vertex_map[v.Id]]))
            edges_index_v_s_in.append(np.array([vertex_map[v.Id], state_map[s]]))

    data[TORCH.game_vertex].x = torch.tensor(np.array(nodes_vertex), dtype=torch.float)
    data[TORCH.state_vertex].x = torch.tensor(np.array(nodes_state), dtype=torch.float)
    data[TORCH.path_condition_vertex].x = torch.tensor(
        np.array(nodes_path_condition), dtype=torch.float
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

StatesDistribution: TypeAlias = torch.tensor


def get_states_distribution(
    file_path: str, state_map: StateMap
) -> StatesDistribution:
    states_distribution = torch.zeros([len(state_map.keys()), 1])
    f = open(file_path)
    state_id = int(f.read())
    f.close()
    states_distribution[state_map[state_id]] = 1
    return states_distribution

@dataclass
class Step:
    graph: HeteroData
    states_distribution: StatesDistribution
    
import torch
import numpy as np

import numpy as np
import torch
from torch_geometric.data import HeteroData
#from ml.inference import TORCH


def remove_path_condition_root(hetero_data: HeteroData) -> HeteroData:
    """
    Removes the PathConditionRoot node from Heterodata and creates edges 
    between states and pathCondition instead.
    
    Parameters
    ----------
    hetero_data : HeteroData
        Heterodata to process.
        
    Returns
    -------
    HeteroData
        New HeteroData without PathConditionRoot.
    """
    new_heterodata = hetero_data.clone()
    
    path_condition_vertex = new_heterodata[TORCH.path_condition_vertex].x.detach().cpu().numpy()
    pathcondvertex_to_pathcondvertex = new_heterodata[TORCH.pathcondvertex_to_pathcondvertex].edge_index.detach().cpu().numpy()
    statevertex_to_pathcondvertex = new_heterodata[TORCH.statevertex_to_pathcondvertex].edge_index.detach().cpu().numpy()
    
    nodes_path_condition = []
    edge_index_pc_pc, edge_index_pc_state, edge_index_state_pc = [], [], []


    for vector in path_condition_vertex:
        new_vector = vector[:-1]

        if vector[-1] != 1.:
            nodes_path_condition.append(new_vector)

    i = 0
    for pc_v in pathcondvertex_to_pathcondvertex[0]:
        if pathcondvertex_to_pathcondvertex[0][i] not in statevertex_to_pathcondvertex[1] and \
            pathcondvertex_to_pathcondvertex[1][i] not in statevertex_to_pathcondvertex[1]:

            k1 = 0
            while k1 < len(statevertex_to_pathcondvertex[1]) and pathcondvertex_to_pathcondvertex[0][i] > statevertex_to_pathcondvertex[1][k1]:
                k1 += 1
            
            k2 = 0
            while k2 < len(statevertex_to_pathcondvertex[1]) and pathcondvertex_to_pathcondvertex[1][i] > statevertex_to_pathcondvertex[1][k2]:
                k2 += 1

            edge_index_pc_pc.extend(
                [
                    [pathcondvertex_to_pathcondvertex[0][i] - k1, pathcondvertex_to_pathcondvertex[1][i] - k2],
                ]
            )  
        i += 1    

    for pc_v in statevertex_to_pathcondvertex[1]:
        ind = np.where(pathcondvertex_to_pathcondvertex[0] == pc_v)[0]
        i = np.where(statevertex_to_pathcondvertex[1] == pc_v)[0][0]

        for pc_v in ind:

            k = 0
            while pathcondvertex_to_pathcondvertex[0][pc_v + 1] > statevertex_to_pathcondvertex[1][k]:
                k += 1

            edge_index_pc_state.append(
                [pathcondvertex_to_pathcondvertex[0][pc_v + 1] - k, statevertex_to_pathcondvertex[0][i]]
            )
            edge_index_state_pc.append(
                [statevertex_to_pathcondvertex[0][i], pathcondvertex_to_pathcondvertex[0][pc_v + 1] - k]
            )

    def null_if_empty(tensor):
        return tensor if tensor.numel() != 0 else torch.empty((2, 0), dtype=torch.int64)
    
    new_heterodata[TORCH.path_condition_vertex].x = torch.tensor(np.array(nodes_path_condition), dtype=torch.float)
    
    new_heterodata[TORCH.pathcondvertex_to_pathcondvertex].edge_index = null_if_empty(
        torch.tensor(np.array(edge_index_pc_pc), dtype=torch.long).t().contiguous())
    
    new_heterodata[TORCH.pathcondvertex_to_statevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edge_index_pc_state), dtype=torch.long).t().contiguous())
    
    new_heterodata[TORCH.statevertex_to_pathcondvertex].edge_index = null_if_empty(
        torch.tensor(np.array(edge_index_state_pc), dtype=torch.long).t().contiguous())
    
    return new_heterodata


# json_data_1 = Path("C:\\Users\\dvtit\\Desktop\\PySymGym\\resources\\onnx\\reference_gamestates\\86_gameState.json")
# json_data_2 = Path("C:\\Users\\dvtit\\Desktop\\PySymGym\\resources\\onnx\\reference_gamestates\\4009_gameState.json")
# json_data_3 = Path("C:\\Users\\dvtit\\Desktop\\PySymGym\\resources\\onnx\\reference_gamestates\\4036_gameState.json")
# json_data_4 = Path("C:\\Users\\dvtit\\Desktop\\PySymGym\\resources\\onnx\\reference_gamestates\\4781_gameState.json")


# heterodata1 = get_heterodata_from_json(json_data_1)
# heterodata2 = get_heterodata_from_json(json_data_2)
# heterodata3 = get_heterodata_from_json(json_data_3)
# heterodata4 = get_heterodata_from_json(json_data_4)

# new_heterodata1 = remove_path_condition_root(heterodata1)
# new_heterodata2 = remove_path_condition_root(heterodata2)
# new_heterodata3 = remove_path_condition_root(heterodata3)
# new_heterodata4 = remove_path_condition_root(heterodata4)

expected1 = HeteroData()

expected1['game_vertex'].x = torch.tensor([
    [1., 2., 1., 0., 0., 1., 1.],
    [1., 6., 1., 0., 1., 1., 1.],
    [1., 1., 1., 0., 0., 1., 1.],
    [1., 3., 1., 0., 0., 1., 1.],
    [1., 6., 1., 0., 1., 1., 1.],
    [1., 1., 1., 0., 0., 1., 1.],
    [1., 3., 1., 0., 0., 1., 1.],
    [1., 6., 1., 0., 1., 1., 1.],
    [1., 1., 1., 0., 0., 1., 1.],
    [1., 9., 1., 0., 0., 1., 1.],
    [1., 5., 1., 0., 0., 1., 1.],
    [1., 2., 1., 0., 0., 1., 1.],
    [1., 5., 1., 0., 0., 0., 1.],
    [1., 4., 1., 0., 0., 0., 1.],
    [1., 4., 1., 0., 0., 0., 1.],
    [1., 3., 1., 0., 0., 1., 1.],
    [1., 2., 1., 0., 0., 1., 1.]
])

expected1['state_vertex'].x = torch.tensor([
    [5., 0., 0., 0., 90., 31.],
    [4., 0., 0., 0., 83., 35.],
    [2., 0., 0., 0., 72., 29.],
    [2., 0., 0., 0., 87., 28.]
])

expected1['path_condition_vertex'].x = torch.tensor([
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
])

expected1[('game_vertex', 'to', 'game_vertex')].edge_index = torch.tensor([
    [0,  0,  1,  3,  3,  4,  6,  6,  7,  9, 10, 10, 12, 12, 13, 14, 15, 15],
    [1,  3,  2,  4,  6,  5,  7,  9,  8, 15, 11, 12, 13, 14, 15, 15, 16, 10]
])

expected1[('game_vertex', 'to', 'game_vertex')].edge_type = torch.tensor(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
)

expected1[('state_vertex', 'in', 'game_vertex')].edge_index = torch.tensor([
    [0,  1,  2,  3],
    [10, 12, 16, 16]
])

expected1[('game_vertex', 'in', 'state_vertex')].edge_index = torch.tensor([
    [10, 12, 16, 16],
    [0,  1,  2,  3]
])

expected1[('state_vertex', 'history', 'game_vertex')].edge_index = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 3, 6, 9, 15, 10, 12, 13, 0, 3, 6, 9, 15, 10, 12, 14, 0, 3, 6, 9, 15, 10, 12, 14, 16, 0, 3, 6, 9, 15, 10, 12, 13, 16]
])

expected1[('state_vertex', 'history', 'game_vertex')].edge_attr = torch.tensor([
    [1, 0], [1, 3], [1, 8], [1, 14], [2, 78], [2, 85], [1, 47], [1, 57],
    [1, 0], [1, 3], [1, 8], [1, 14], [2, 65], [2, 69], [2, 77], [1, 58],
    [1, 0], [1, 3], [1, 8], [1, 14], [2, 65], [1, 36], [1, 47], [1, 58], [1, 68],
    [1, 0], [1, 3], [1, 8], [1, 14], [2, 78], [1, 36], [1, 47], [1, 57], [1, 84]
])

expected1[('game_vertex', 'history', 'state_vertex')].edge_index = torch.tensor([
    [0, 3, 6, 9, 15, 10, 12, 13, 0, 3, 6, 9, 15, 10, 12, 14, 0, 3, 6, 9, 15, 10, 12, 14, 16, 0, 3, 6, 9, 15, 10, 12, 13, 16],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3]
])

expected1[('game_vertex', 'history', 'state_vertex')].edge_attr = torch.tensor([
    [1, 0], [1, 3], [1, 8], [1, 14], [2, 78], [2, 85], [1, 47], [1, 57], 
    [1, 0], [1, 3], [1, 8], [1, 14], [2, 65], [2, 69], [2, 77], [1, 58], 
    [1, 0], [1, 3], [1, 8], [1, 14], [2, 65], [1, 36], [1, 47], [1, 58], [1, 68], 
    [1, 0], [1, 3], [1, 8], [1, 14], [2, 78], [1, 36], [1, 47], [1, 57], [1, 84]
])

expected1[('state_vertex', 'parent_of', 'state_vertex')].edge_index = torch.tensor([
    [2, 3, 3],
    [1, 2, 0]
])

expected1[('path_condition_vertex', 'to', 'path_condition_vertex')].edge_index = torch.tensor([
    [0, 10, 0, 1, 1, 3, 1, 2, 3, 8, 3, 4, 4, 6, 4, 5, 6, 9, 6, 7, 7, 8, 10, 3, 
     10, 11, 12, 13, 13, 15, 13, 14, 16, 17, 17, 19, 17, 18, 20, 8, 20, 9, 21, 
     8, 21, 3, 22, 23, 23, 8, 23, 11, 24, 25, 25, 8, 25, 9, 26, 19, 26, 18, 27, 
     26, 28, 29, 28, 9, 29, 30, 29, 3, 31, 28, 32, 21],
    [10, 0, 1, 0, 3, 1, 2, 1, 8, 3, 4, 3, 6, 4, 5, 4, 9, 6, 7, 6, 8, 7, 3, 10, 
     11, 10, 13, 12, 15, 13, 14, 13, 17, 16, 19, 17, 18, 17, 8, 20, 9, 20, 8, 
     21, 3, 21, 23, 22, 8, 23, 11, 23, 25, 24, 8, 25, 9, 25, 19, 26, 18, 26, 26, 
     27, 29, 28, 9, 28, 30, 29, 3, 29, 28, 31, 21, 32]
])

expected1[('path_condition_vertex', 'to', 'state_vertex')].edge_index = torch.tensor([
    [0, 12, 16, 20, 21, 22, 24, 26, 0, 12, 16, 20, 27, 28, 22, 24, 0, 12, 16, 20, 27, 31, 22, 24, 0, 12, 16, 20, 22, 24, 32, 26],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
])

expected1[('state_vertex', 'to', 'path_condition_vertex')].edge_index = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 12, 16, 20, 21, 22, 24, 26, 0, 12, 16, 20, 27, 28, 22, 24, 0, 12, 16, 20, 27, 31, 22, 24, 0, 12, 16, 20, 22, 24, 32, 26]
])


expected2 = HeteroData()

expected2['game_vertex'].x = torch.tensor([
    [0., 2., 1., 0., 1., 0., 1.],
    [0., 1., 1., 0., 0., 0., 1.],
    [0., 2., 1., 0., 1., 0., 1.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 2., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 1., 0., 1.],
    [0., 20., 0., 0., 0., 0., 0.],
    [0., 3., 1., 0., 1., 0., 1.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 2., 1., 0., 1., 0., 1.],
    [0., 11., 1., 0., 0., 0., 1.],
    [0., 15., 1., 0., 0., 0., 1.],
    [0., 3., 1., 0., 0., 0., 1.],
    [0., 8., 1., 0., 1., 0., 1.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 4., 0., 0., 0., 0., 0.],
    [0., 8., 1., 0., 0., 0., 1.],
    [0., 4., 1., 0., 0., 0., 1.],
    [0., 10., 1., 0., 0., 0., 0.],
    [0., 8., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 4., 0., 0., 0., 0., 0.],
    [0., 3., 0., 0., 0., 0., 0.],
    [0., 4., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 27., 1., 0., 0., 0., 1.],
    [1., 2., 1., 0., 1., 0., 1.],
    [1., 9., 0., 0., 1., 0., 0.],
    [1., 1., 0., 0., 0., 0., 0.],
    [1., 5., 0., 0., 1., 0., 0.],
    [1., 4., 0., 0., 1., 0., 0.],
    [1., 4., 0., 0., 1., 0., 0.],
    [1., 5., 0., 0., 1., 0., 0.],
    [1., 10., 0., 0., 1., 0., 0.],
    [1., 1., 0., 0., 1., 0., 0.],
    [1., 1., 0., 0., 0., 0., 0.],
    [1., 4., 0., 0., 1., 0., 0.],
    [1., 1., 0., 0., 0., 0., 0.]
])

expected2['state_vertex'].x = torch.tensor([
    [8., 0., 0., 0., 58., 24.],
    [17., 0., 0., 0., 61., 40.]
])

expected2['path_condition_vertex'].x = torch.tensor([
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
])

expected2[('game_vertex', 'to', 'game_vertex')].edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 20, 21, 21, 22, 23, 23, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 31, 32, 33, 34, 35, 37],
    [1, 10, 2, 4, 3, 6, 15, 5, 15, 7, 26, 3, 9, 12, 28, 11, 26, 1, 13, 18, 17, 15, 0, 16, 18, 17, 18, 14, 19, 24, 23, 21, 22, 24, 23, 24, 20, 25, 9, 11, 7, 28, 8, 29, 30, 37, 31, 32, 33, 34, 35, 36, 38]
])

expected2[('game_vertex', 'to', 'game_vertex')].edge_type = torch.tensor(
    [0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
)

expected2[('state_vertex', 'in', 'game_vertex')].edge_index = torch.tensor([
    [0,  1],
    [19, 26]
])

expected2[('game_vertex', 'in', 'state_vertex')].edge_index = torch.tensor([
    [19, 26],
    [ 0,  1]
])

expected2[('state_vertex', 'history', 'game_vertex')].edge_index = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [27, 8, 12, 13, 17, 18, 19, 27, 8, 12, 13, 17, 14, 0, 10, 26, 11, 1, 2, 6]
])

expected2[('state_vertex', 'history', 'game_vertex')].edge_attr = torch.tensor([
    [1, 0], [1, 2], [1, 5], [1, 16], [1, 19], [1, 23], [1, 48], [1, 0], [1, 2], 
    [1, 5], [1, 16], [1, 19], [1, 24], [1, 29], [1, 31], [2, 55], [1, 42], [1, 50], [1, 51], [1, 54]
])

expected2[('game_vertex', 'history', 'state_vertex')].edge_index = torch.tensor([
    [27, 8, 12, 13, 17, 18, 19, 27, 8, 12, 13, 17, 14, 0, 10, 26, 11, 1, 2, 6],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

expected2[('game_vertex', 'history', 'state_vertex')].edge_attr = torch.tensor([
    [1, 0], [1, 2], [1, 5], [1, 16], [1, 19], [1, 23], [1, 48], [1, 0], [1, 2], 
    [1, 5], [1, 16], [1, 19], [1, 24], [1, 29], [1, 31], [2, 55], [1, 42], [1, 50], [1, 51], [1, 54]
])

expected2[('state_vertex', 'parent_of', 'state_vertex')].edge_index = torch.tensor([
    [0],
    [1]
])

expected2[('path_condition_vertex', 'to', 'path_condition_vertex')].edge_index = torch.tensor([
    [0, 1, 1, 3, 1, 2, 4, 5, 5, 7, 5, 6, 8, 2, 8, 3],
    [1, 0, 3, 1, 2, 1, 5, 4, 7, 5, 6, 5, 2, 8, 3, 8]
])

expected2[('path_condition_vertex', 'to', 'state_vertex')].edge_index = torch.tensor([
    [0, 4, 1, 4, 8],
    [0, 0, 1, 1, 1]
])

expected2[('state_vertex', 'to', 'path_condition_vertex')].edge_index = torch.tensor([
    [0, 0, 1, 1, 1],
    [0, 4, 1, 4, 8]
])


expected3 = HeteroData()

expected3['game_vertex'].x = torch.tensor([
    [0., 2., 1., 0., 1., 0., 1.],
    [0., 1., 1., 0., 0., 0., 1.],
    [0., 2., 1., 0., 1., 0., 1.],
    [0., 1., 1., 0., 0., 0., 1.],
    [0., 2., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 1., 0., 1.],
    [0., 20., 1., 0., 0., 0., 1.],
    [0., 3., 1., 0., 1., 0., 1.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 2., 1., 0., 1., 0., 1.],
    [0., 11., 1., 0., 0., 0., 1.],
    [0., 15., 1., 0., 0., 0., 1.],
    [0., 3., 1., 0., 0., 0., 1.],
    [0., 8., 1., 0., 1., 0., 1.],
    [0., 1., 1., 0., 0., 0., 1.],
    [0., 4., 0., 0., 0., 0., 0.],
    [0., 8., 1., 0., 0., 0., 1.],
    [0., 4., 1., 0., 0., 0., 1.],
    [0., 10., 1., 0., 0., 0., 1.],
    [0., 8., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 4., 0., 0., 0., 0., 0.],
    [0., 3., 1., 0., 0., 0., 1.],
    [0., 4., 1., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 27., 1., 0., 0., 0., 1.],
    [1., 2., 1., 0., 1., 0., 1.],
    [1., 9., 0., 0., 1., 0., 0.],
    [1., 1., 0., 0., 0., 0., 0.],
    [1., 5., 0., 0., 1., 0., 0.],
    [1., 4., 0., 0., 1., 0., 0.],
    [1., 4., 0., 0., 1., 0., 0.],
    [1., 5., 0., 0., 1., 0., 0.],
    [1., 10., 0., 0., 1., 0., 0.],
    [1., 1., 0., 0., 1., 0., 0.],
    [1., 1., 0., 0., 0., 0., 0.],
    [1., 4., 0., 0., 1., 0., 0.],
    [1., 1., 0., 0., 0., 0., 0.]
])

expected3['state_vertex'].x = torch.tensor([
    [4., 0., 0., 0., 87., 57.],
    [1., 0., 0., 0., 88., 28.]
])

expected3['path_condition_vertex'].x = torch.tensor([
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
])

expected3[('game_vertex', 'to', 'game_vertex')].edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 20, 21, 21, 22, 23, 23, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 31, 32, 33, 34, 35, 37],
    [1, 10, 2, 4, 3, 6, 15, 5, 15, 7, 26, 3, 9, 12, 28, 11, 26, 1, 13, 18, 17, 15, 0, 16, 18, 17, 18, 14, 19, 24, 23, 21, 22, 24, 23, 24, 20, 25, 9, 11, 7, 28, 8, 29, 30, 37, 31, 32, 33, 34, 35, 36, 38]
])

expected3[('game_vertex', 'to', 'game_vertex')].edge_type = torch.tensor(
    [0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
)

expected3[('state_vertex', 'in', 'game_vertex')].edge_index = torch.tensor([
    [0,  1],
    [18, 24]
])

expected3[('game_vertex', 'in', 'state_vertex')].edge_index = torch.tensor([
    [18, 24],
    [ 0,  1]
])

expected3[('state_vertex', 'history', 'game_vertex')].edge_index = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [27, 8, 12, 13, 17, 14, 0, 10, 26, 11, 1, 2, 6, 7, 3, 15, 18, 27, 8, 12, 13, 17, 18, 19, 23, 24]
])

expected3[('state_vertex', 'history', 'game_vertex')].edge_attr = torch.tensor([
    [1, 0], [1, 2], [1, 5], [1, 16], [1, 19], [1, 24], [1, 29], [1, 31], [2, 55], 
    [1, 42], [1, 50], [1, 51], [1, 54], [1, 67], [1, 78], [1, 81], [1, 83], [1, 0], 
    [1, 2], [1, 5], [1, 16], [1, 19], [1, 23], [1, 48], [1, 80], [1, 88]
])

expected3[('game_vertex', 'history', 'state_vertex')].edge_index = torch.tensor([
    [27, 8, 12, 13, 17, 14, 0, 10, 26, 11, 1, 2, 6, 7, 3, 15, 18, 27, 8, 12, 13, 17, 18, 19, 23, 24],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

expected3[('game_vertex', 'history', 'state_vertex')].edge_attr = torch.tensor([
    [1, 0], [1, 2], [1, 5], [1, 16], [1, 19], [1, 24], [1, 29], [1, 31], [2, 55], 
    [1, 42], [1, 50], [1, 51], [1, 54], [1, 67], [1, 78], [1, 81], [1, 83], [1, 0], 
    [1, 2], [1, 5], [1, 16], [1, 19], [1, 23], [1, 48], [1, 80], [1, 88]
])

expected3[('state_vertex', 'parent_of', 'state_vertex')].edge_index = torch.tensor([
    [1],
    [0]
])

expected3[('path_condition_vertex', 'to', 'path_condition_vertex')].edge_index = torch.tensor([
    [0, 2, 0, 1, 3, 4, 4, 6, 4, 5, 7, 1, 7, 2, 8, 0],
    [2, 0, 1, 0, 4, 3, 6, 4, 5, 4, 1, 7, 2, 7, 0, 8]
])

expected3[('path_condition_vertex', 'to', 'state_vertex')].edge_index = torch.tensor([
    [0, 3, 7, 8, 3],
    [0, 0, 0, 1, 1]
])

expected3[('state_vertex', 'to', 'path_condition_vertex')].edge_index = torch.tensor([
    [0, 0, 0, 1, 1],
    [0, 3, 7, 8, 3]
])


expected4 = HeteroData()

expected4['game_vertex'].x = torch.tensor([
    [1., 9., 1., 0., 1., 0., 0.],
    [1., 4., 0., 0., 0., 0., 0.]
])

expected4['state_vertex'].x = torch.tensor([
    [1., 0., 0., 0., 0., 0.]
])

expected4['path_condition_vertex'].x = torch.tensor([
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
])

expected4[('game_vertex', 'to', 'game_vertex')].edge_index = torch.tensor([
    [0],
    [1]
])

expected4[('game_vertex', 'to', 'game_vertex')].edge_type = torch.tensor(
    [0]
)

expected4[('state_vertex', 'in', 'game_vertex')].edge_index = torch.tensor([
    [0],
    [0]
])

expected4[('game_vertex', 'in', 'state_vertex')].edge_index = torch.tensor([
    [0],
    [0]
])

expected4[('state_vertex', 'history', 'game_vertex')].edge_index = torch.tensor([
    [0],
    [0]
])

expected4[('state_vertex', 'history', 'game_vertex')].edge_attr = torch.tensor([
    [1, 0]
])

expected4[('game_vertex', 'history', 'state_vertex')].edge_index = torch.tensor([
    [0],
    [0]
])

expected4[('game_vertex', 'history', 'state_vertex')].edge_attr = torch.tensor([
    [1, 0]
])

expected4[('state_vertex', 'parent_of', 'state_vertex')].edge_index = torch.empty((2, 0), dtype=torch.long)

expected4[('path_condition_vertex', 'to', 'path_condition_vertex')].edge_index = torch.tensor([
    [0, 1, 1, 3, 1, 2],
    [1, 0, 3, 1, 2, 1]
])

expected4[('path_condition_vertex', 'to', 'state_vertex')].edge_index = torch.tensor([
    [0],
    [0]
])

expected4[('state_vertex', 'to', 'path_condition_vertex')].edge_index = torch.tensor([
    [0],
    [0]
])


class EdgeTypes:
    g_t_g = ('game_vertex', 'to', 'game_vertex')
    s_i_g = ('state_vertex', 'in', 'game_vertex')
    g_i_s = ('game_vertex', 'in', 'state_vertex')
    s_h_g = ('state_vertex', 'history', 'game_vertex')
    g_h_s = ('game_vertex', 'history', 'state_vertex')
    s_p_s = ('state_vertex', 'parent_of', 'state_vertex')
    p_t_p = ('path_condition_vertex', 'to', 'path_condition_vertex')
    p_t_s = ('path_condition_vertex', 'to', 'state_vertex')
    s_t_p = ('state_vertex', 'to', 'path_condition_vertex')
    

# def compare(h1: HeteroData, h2: HeteroData):
#     assert torch.equal(h1[TORCH.game_vertex].x, h2[TORCH.game_vertex].x)
#     assert torch.equal(h1[TORCH.state_vertex].x, h2[TORCH.state_vertex].x)
#     assert torch.equal(h1[TORCH.path_condition_vertex].x, h2[TORCH.path_condition_vertex].x)

#     assert torch.equal(h1[EdgeTypes.g_t_g].edge_index, h2[EdgeTypes.g_t_g].edge_index)
#     assert torch.equal(h1[EdgeTypes.g_t_g].edge_type, h2[EdgeTypes.g_t_g].edge_type)

#     assert torch.equal(h1[EdgeTypes.s_i_g].edge_index, h2[EdgeTypes.s_i_g].edge_index)

#     assert torch.equal(h1[EdgeTypes.g_i_s].edge_index, h2[EdgeTypes.g_i_s].edge_index)

#     assert torch.equal(h1[EdgeTypes.s_h_g].edge_index, h2[EdgeTypes.s_h_g].edge_index)
#     assert torch.equal(h1[EdgeTypes.s_h_g].edge_attr, h2[EdgeTypes.s_h_g].edge_attr)

#     assert torch.equal(h1[EdgeTypes.g_h_s].edge_index, h2[EdgeTypes.g_h_s].edge_index)
#     assert torch.equal(h1[EdgeTypes.g_h_s].edge_attr, h2[EdgeTypes.g_h_s].edge_attr)

#     assert torch.equal(h1[EdgeTypes.s_p_s].edge_index, h2[EdgeTypes.s_p_s].edge_index)
#     assert torch.equal(h1[EdgeTypes.p_t_p].edge_index, h2[EdgeTypes.p_t_p].edge_index)
#     assert torch.equal(h1[EdgeTypes.p_t_s].edge_index, h2[EdgeTypes.p_t_s].edge_index)
#     assert torch.equal(h1[EdgeTypes.s_t_p].edge_index, h2[EdgeTypes.s_t_p].edge_index)

# def non_all_features_are_zeros(h: HeteroData):
#     for vector in h[TORCH.game_vertex].x.detach().cpu().numpy():
#         if not np.all(vector != 0):
#             return False

# compare(new_heterodata1, expected1)
# compare(new_heterodata2, expected2)
# compare(new_heterodata3, expected3)
# compare(new_heterodata4, expected4)

# non_all_features_are_zeros(new_heterodata1)
# non_all_features_are_zeros(new_heterodata2)
# non_all_features_are_zeros(new_heterodata3)
# non_all_features_are_zeros(new_heterodata4)

def get_heterodata_from_json(path: Path):
    with open(path, 'r') as f:
        json1 = json.load(f)
    game_state = GameState.from_dict(json1)
    heterodata, state_map = convert_input_to_tensor(game_state)
    return heterodata

@pytest.fixture
def test_data():
    json_data_1 = Path("C:\\Users\\dvtit\\Desktop\\PySymGym\\resources\\onnx\\reference_gamestates\\86_gameState.json")
    json_data_2 = Path("C:\\Users\\dvtit\\Desktop\\PySymGym\\resources\\onnx\\reference_gamestates\\4009_gameState.json")
    json_data_3 = Path("C:\\Users\dvtit\\Desktop\\PySymGym\\resources\\onnx\\reference_gamestates\\4036_gameState.json")
    json_data_4 = Path("C:\\Users\dvtit\\Desktop\\PySymGym\\resources\\onnx\\reference_gamestates\\4781_gameState.json")
    
    heterodata1 = get_heterodata_from_json(json_data_1)
    heterodata2 = get_heterodata_from_json(json_data_2)
    heterodata3 = get_heterodata_from_json(json_data_3)
    heterodata4 = get_heterodata_from_json(json_data_4)
    
    return [
        (heterodata1, expected1),
        (heterodata2, expected2),
        (heterodata3, expected3),
        (heterodata4, expected4),
    ]

def test_compare_all(test_data):
    for result1, expected in test_data:
        result = remove_path_condition_root(result1)
        assert torch.equal(result[TORCH.game_vertex].x, expected[TORCH.game_vertex].x)
        assert torch.equal(result[TORCH.state_vertex].x, expected[TORCH.state_vertex].x)
        assert torch.equal(result[TORCH.path_condition_vertex].x, expected[TORCH.path_condition_vertex].x)
        assert torch.equal(result[EdgeTypes.g_t_g].edge_index, expected[EdgeTypes.g_t_g].edge_index)
        assert torch.equal(result[EdgeTypes.g_t_g].edge_type, expected[EdgeTypes.g_t_g].edge_type)
        assert torch.equal(result[EdgeTypes.s_i_g].edge_index, expected[EdgeTypes.s_i_g].edge_index)
        assert torch.equal(result[EdgeTypes.g_i_s].edge_index, expected[EdgeTypes.g_i_s].edge_index)
        assert torch.equal(result[EdgeTypes.s_h_g].edge_index, expected[EdgeTypes.s_h_g].edge_index)
        assert torch.equal(result[EdgeTypes.s_h_g].edge_attr, expected[EdgeTypes.s_h_g].edge_attr)
        assert torch.equal(result[EdgeTypes.g_h_s].edge_index, expected[EdgeTypes.g_h_s].edge_index)
        assert torch.equal(result[EdgeTypes.g_h_s].edge_attr, expected[EdgeTypes.g_h_s].edge_attr)
        assert torch.equal(result[EdgeTypes.s_p_s].edge_index, expected[EdgeTypes.s_p_s].edge_index)
        assert torch.equal(result[EdgeTypes.p_t_p].edge_index, expected[EdgeTypes.p_t_p].edge_index)
        assert torch.equal(result[EdgeTypes.p_t_s].edge_index, expected[EdgeTypes.p_t_s].edge_index)
        assert torch.equal(result[EdgeTypes.s_t_p].edge_index, expected[EdgeTypes.s_t_p].edge_index)

def test_non_all_features_are_zeros(test_data):
    for result1, _ in test_data:
        result = remove_path_condition_root(result1)
        for vector in result[TORCH.game_vertex].x.detach().cpu().numpy():
            assert not np.all(vector == 0)