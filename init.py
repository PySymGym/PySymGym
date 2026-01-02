from dataclasses import dataclass

#from common.validation_coverage.svm_info import SVMInfo
from dataclasses_json import dataclass_json

import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Tuple,
    TypeAlias,
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



from torch_geometric.data import Dataset, HeteroData

GAMESTATESUFFIX = "_gameState"
STATESUFFIX = "_statesInfo"
MOVEDSTATESUFFIX = "_movedState"

NUM_PC_FEATURES = 49
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




def pc_remover (hetero_data: HeteroData) -> HeteroData:
    """
    Removes the PathConditionRoot node from Heterodata from disk and creates edges between states and pathCondition instead.
    
    Parameters
    ----------
    hetero_data : HeteroData
        Heterodata from disk.
    save_path : Path
        The path where the new HeteroData will be saved without PathConditionRoot.
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
            
            edge_index_pc_pc.extend(
                [
                    [pathcondvertex_to_pathcondvertex[0][i], pathcondvertex_to_pathcondvertex[1][i]],
                ]
            )    
        i += 1   
    
    for pc_v in statevertex_to_pathcondvertex[1]:
        ind = np.where(pathcondvertex_to_pathcondvertex[0] == pc_v)[0]
        i = np.where(statevertex_to_pathcondvertex[1] == pc_v)[0][0]

        for pc_v in ind:
            edge_index_pc_state.append(
                [pathcondvertex_to_pathcondvertex[0][pc_v+1], statevertex_to_pathcondvertex[0][i]]
            )
            edge_index_state_pc.append(
                [statevertex_to_pathcondvertex[0][i], pathcondvertex_to_pathcondvertex[0][pc_v+1]]
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



def main():


    json_data_1 = Path("/home/banicsidy/Desktop/PySymGym/resources/onnx/reference_gamestates/86_gameState.json")

    with open(json_data_1, 'r') as f:
        json1 = json.load(f)
    game_state = GameState.from_dict(json1)
    heterodata, state_map = convert_input_to_tensor(game_state)

    new_heterodata1 = pc_remover(heterodata)


    print(heterodata)
    print(new_heterodata1)

    # print(heterodata[TORCH.path_condition_vertex].x.detach().cpu().numpy())
    # print()
    # print(new_heterodata1[TORCH.path_condition_vertex].x.detach().cpu().numpy())
    # print(heterodata[TORCH.statevertex_to_pathcondvertex].edge_index.detach().cpu().numpy())
    # print(new_heterodata1[TORCH.statevertex_to_pathcondvertex].edge_index.detach().cpu().numpy())
    print(heterodata[TORCH.pathcondvertex_to_pathcondvertex].edge_index.detach().cpu().numpy())
    print(new_heterodata1[TORCH.pathcondvertex_to_pathcondvertex].edge_index.detach().cpu().numpy())
 

    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx

    graph = to_networkx(new_heterodata1, to_undirected=False)

    node_type_colors = {
        "game_vertex": "#EDEDED00",
        "state_vertex": "#ED8546",
        "path_condition_vertex": "#70B349",
    }

    edge_type_colors = {
        ("game_vertex", "to", "game_vertex"): "#8B4D9E00",
        ("state_vertex", "in", "game_vertex"): "#DFB72500",
        ("game_vertex", "in", "state_vertex"): "#8B4D9E00",
        ("state_vertex", "history", "game_vertex"): "#DB5C6400",
        ("game_vertex", "history", "state_vertex"): "#DB5C6400",
        ("state_vertex", "parent_of", "state_vertex"): "#4599C300",
        ("path_condition_vertex", "to", "path_condition_vertex"): "#6AB83DFF",
        ("path_condition_vertex", "to", "state_vertex"): "#ED8546",
        ("state_vertex", "to", "path_condition_vertex"): "#ED8546",
    }

    node_colors = []
    labels = {}
    print(graph.nodes(data=True))
    for node, attrs in graph.nodes(data=True):
        print(node, attrs)
        node_type = attrs["type"]
        color = node_type_colors[node_type]
        node_colors.append(color)
        
        # if node_type == "game_vertex":
        #     labels[node] = f"G{node}"
        if node_type == "state_vertex":
            labels[node] = f"S{node}"
        elif node_type == "path_condition_vertex":
            labels[node] = f"PC{node}"
        # else:
        #     labels[node] = f"N{node}"

    edge_colors = []
    for from_node, to_node, attrs in graph.edges(data=True):
        edge_type = attrs["type"]
        color = edge_type_colors[edge_type]
        graph.edges[from_node, to_node]["color"] = color
        edge_colors.append(color)

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph, k=2, seed=42)

    nx.draw_networkx(
        graph,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=500,
        font_size=10,
        arrows=True,
        arrowsize=15,
    )

    import matplotlib.patches as mpatches
    legend_patches = []
    for node_type, color in node_type_colors.items():
        legend_patches.append(mpatches.Patch(color=color, label=node_type))

    plt.legend(handles=legend_patches, loc='upper right')
    plt.title("Heterodata with pc_root")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Граф сохранен в 'graph_visualization.png'")


if __name__ == "__main__":
    main()