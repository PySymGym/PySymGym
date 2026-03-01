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

    edge_index_pc_pc, edge_index_pc_state, edge_index_state_pc = [], [], []

    path_condition_vertex = path_condition_vertex[path_condition_vertex[:, -1] != 1.0][
        :, :-1
    ]

    def index_corrector(el):
        shift = 0
        while (
            shift < len(statevertex_to_pathcondvertex[1])
            and el > statevertex_to_pathcondvertex[1][shift]
        ):
            shift += 1
        return shift

    for pc_vertex_from, pc_vertex_to in zip(*pathcondvertex_to_pathcondvertex):
        if (
            pc_vertex_from not in statevertex_to_pathcondvertex[1]
            and pc_vertex_to not in statevertex_to_pathcondvertex[1]
        ):
            shift1 = index_corrector(pc_vertex_from)

            shift2 = index_corrector(pc_vertex_to)

            edge_index_pc_pc.extend(
                [
                    [
                        pc_vertex_from - shift1,
                        pc_vertex_to - shift2,
                    ],
                ]
            )

    for _root_in_states in statevertex_to_pathcondvertex[1]:
        index_of_roots_in_pathcond_to_pathcond = np.where(
            pathcondvertex_to_pathcondvertex[0] == _root_in_states
        )[0]
        indexes_of_roots_in_states = np.where(
            statevertex_to_pathcondvertex[1] == _root_in_states
        )[0][0]

        for index in index_of_roots_in_pathcond_to_pathcond:
            pc_vertex = pathcondvertex_to_pathcondvertex[0][index + 1]
            state_id = statevertex_to_pathcondvertex[0][indexes_of_roots_in_states]

            shift = index_corrector(pc_vertex)

            edge_index_pc_state.append(
                [
                    pc_vertex - shift,
                    state_id,
                ]
            )

            edge_index_state_pc.append(
                [
                    state_id,
                    pc_vertex - shift,
                ]
            )

    def null_if_empty(tensor):
        return tensor if tensor.numel() != 0 else torch.empty((2, 0), dtype=torch.int64)

    new_heterodata[TORCH.path_condition_vertex].x = torch.tensor(
        path_condition_vertex, dtype=torch.float
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
