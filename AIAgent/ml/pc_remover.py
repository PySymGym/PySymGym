import numpy as np
import torch
from torch_geometric.data import HeteroData
from ml.inference import TORCH


def remove_path_condition_root(heterodata: HeteroData) -> HeteroData:
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
    new_heterodata = heterodata.clone()

    path_condition_vertex = (
        new_heterodata[TORCH.path_condition_vertex].x.detach().cpu().numpy()
    )
    pathcondvertex_to_pathcondvertex = (
        new_heterodata[TORCH.pathcondvertex_to_pathcondvertex]
        .edge_index.detach()
        .cpu()
        .numpy()
    )
    statevertex_to_pathcondvertex = (
        new_heterodata[TORCH.statevertex_to_pathcondvertex]
        .edge_index.detach()
        .cpu()
        .numpy()
    )

    map_of_id = []
    pc_x = []
    edge_index_pc_pc, edge_index_pc_state, edge_index_state_pc = [], [], []

    roots_num = 0
    for i, vector in enumerate(path_condition_vertex):
        if vector[-1] != 1.0:
            pc_x.append(vector[:-1])
            map_of_id.append([i, i - roots_num])
        else:
            map_of_id.append([i, -1])
            roots_num += 1

    for root_in_states in statevertex_to_pathcondvertex[1]:
        indexes_of_each_root_in_pathcond_to_pathcond = np.where(
            pathcondvertex_to_pathcondvertex[0] == root_in_states
        )[0]
        index_of_root_in_states = np.where(
            statevertex_to_pathcondvertex[1] == root_in_states
        )[0][0]

        for index in indexes_of_each_root_in_pathcond_to_pathcond:
            child_id = pathcondvertex_to_pathcondvertex[1][index]
            state_id = statevertex_to_pathcondvertex[0][index_of_root_in_states]

            edge_index_pc_state.append(
                [
                    map_of_id[child_id][1],
                    state_id,
                ]
            )

            edge_index_state_pc.append([state_id, map_of_id[child_id][1]])

    for pc_vertex_from, pc_vertex_to in zip(
        *pathcondvertex_to_pathcondvertex, strict=False
    ):
        if map_of_id[pc_vertex_from][1] != -1 and map_of_id[pc_vertex_to][1] != -1:
            edge_index_pc_pc.append(
                [map_of_id[pc_vertex_from][1], map_of_id[pc_vertex_to][1]]
            )

    print("path_condition_vertex old:")
    print("num of features: ", len(path_condition_vertex[1]))
    print("num of vertices: ", len(path_condition_vertex))

    print("path_condition_vertex new:")
    print("num of features: ", len(pc_x[1]))
    print("num of vertices: ", len(pc_x))

    print("pathcondvertex_to_pathcondvertex old:")
    print("max el: ", max(pathcondvertex_to_pathcondvertex[1]))
    print("min el: ", min(pathcondvertex_to_pathcondvertex[1]))

    print("pathcondvertex_to_pathcondvertex new:")
    print("max el: ", max(edge_index_pc_pc[1]))
    print("min el: ", min(edge_index_pc_pc[1]))

    print("edge_index_state_pc old:")
    print("max el: ", max(statevertex_to_pathcondvertex[1]))
    print("min el: ", min(statevertex_to_pathcondvertex[1]))

    print("edge_index_state_pc new:")
    print("max el: ", max(edge_index_state_pc[1]))
    print("min el: ", min(edge_index_state_pc[1]))

    def null_if_empty(tensor):
        return tensor if tensor.numel() != 0 else torch.empty((2, 0), dtype=torch.int64)

    new_heterodata[TORCH.path_condition_vertex].x = torch.tensor(
        pc_x, dtype=torch.float
    )

    new_heterodata[TORCH.pathcondvertex_to_pathcondvertex].edge_index = null_if_empty(
        torch.tensor(edge_index_pc_pc, dtype=torch.long).t().contiguous()
    )

    new_heterodata[TORCH.pathcondvertex_to_statevertex].edge_index = null_if_empty(
        torch.tensor(edge_index_pc_state, dtype=torch.long).t().contiguous()
    )

    new_heterodata[TORCH.statevertex_to_pathcondvertex].edge_index = null_if_empty(
        torch.tensor(edge_index_state_pc, dtype=torch.long).t().contiguous()
    )

    return new_heterodata
