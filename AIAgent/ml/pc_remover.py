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

    nodes_path_condition = []
    edge_index_pc_pc, edge_index_pc_state, edge_index_state_pc = [], [], []

    for vector in path_condition_vertex:
        new_vector = vector[:-1]

        if vector[-1] != 1.0:
            nodes_path_condition.append(new_vector)

    i = 0
    for _pc_v in pathcondvertex_to_pathcondvertex[0]:
        if (
            pathcondvertex_to_pathcondvertex[0][i]
            not in statevertex_to_pathcondvertex[1]
            and pathcondvertex_to_pathcondvertex[1][i]
            not in statevertex_to_pathcondvertex[1]
        ):
            k1 = 0
            while (
                k1 < len(statevertex_to_pathcondvertex[1])
                and pathcondvertex_to_pathcondvertex[0][i]
                > statevertex_to_pathcondvertex[1][k1]
            ):
                k1 += 1

            k2 = 0
            while (
                k2 < len(statevertex_to_pathcondvertex[1])
                and pathcondvertex_to_pathcondvertex[1][i]
                > statevertex_to_pathcondvertex[1][k2]
            ):
                k2 += 1

            edge_index_pc_pc.extend(
                [
                    [
                        pathcondvertex_to_pathcondvertex[0][i] - k1,
                        pathcondvertex_to_pathcondvertex[1][i] - k2,
                    ],
                ]
            )
        i += 1

    for _pc_v in statevertex_to_pathcondvertex[1]:
        ind = np.where(pathcondvertex_to_pathcondvertex[0] == _pc_v)[0]
        i = np.where(statevertex_to_pathcondvertex[1] == _pc_v)[0][0]

        for _pc_v in ind:
            k = 0
            while (
                pathcondvertex_to_pathcondvertex[0][_pc_v + 1]
                > statevertex_to_pathcondvertex[1][k]
            ):
                k += 1

            edge_index_pc_state.append(
                [
                    pathcondvertex_to_pathcondvertex[0][_pc_v + 1] - k,
                    statevertex_to_pathcondvertex[0][i],
                ]
            )
            edge_index_state_pc.append(
                [
                    statevertex_to_pathcondvertex[0][i],
                    pathcondvertex_to_pathcondvertex[0][_pc_v + 1] - k,
                ]
            )

    def null_if_empty(tensor):
        return tensor if tensor.numel() != 0 else torch.empty((2, 0), dtype=torch.int64)

    new_heterodata[TORCH.path_condition_vertex].x = torch.tensor(
        np.array(nodes_path_condition), dtype=torch.float
    )

    new_heterodata[TORCH.pathcondvertex_to_pathcondvertex].edge_index = null_if_empty(
        torch.tensor(np.array(edge_index_pc_pc), dtype=torch.long).t().contiguous()
    )

    new_heterodata[TORCH.pathcondvertex_to_statevertex].edge_index = null_if_empty(
        torch.tensor(np.array(edge_index_pc_state), dtype=torch.long).t().contiguous()
    )

    new_heterodata[TORCH.statevertex_to_pathcondvertex].edge_index = null_if_empty(
        torch.tensor(np.array(edge_index_state_pc), dtype=torch.long).t().contiguous()
    )

    return new_heterodata
