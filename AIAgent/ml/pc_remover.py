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

    for indexes_of_roots_in_states, _ in enumerate(pathcondvertex_to_pathcondvertex[0]):
        if (
            pathcondvertex_to_pathcondvertex[0][indexes_of_roots_in_states]
            not in statevertex_to_pathcondvertex[1]
            and pathcondvertex_to_pathcondvertex[1][indexes_of_roots_in_states]
            not in statevertex_to_pathcondvertex[1]
        ):
            shift1 = 0
            while (
                shift1 < len(statevertex_to_pathcondvertex[1])
                and pathcondvertex_to_pathcondvertex[0][indexes_of_roots_in_states]
                > statevertex_to_pathcondvertex[1][shift1]
            ):
                shift1 += 1

            shift2 = 0
            while (
                shift2 < len(statevertex_to_pathcondvertex[1])
                and pathcondvertex_to_pathcondvertex[1][indexes_of_roots_in_states]
                > statevertex_to_pathcondvertex[1][shift2]
            ):
                shift2 += 1

            edge_index_pc_pc.extend(
                [
                    [
                        pathcondvertex_to_pathcondvertex[0][indexes_of_roots_in_states]
                        - shift1,
                        pathcondvertex_to_pathcondvertex[1][indexes_of_roots_in_states]
                        - shift2,
                    ],
                ]
            )

    for _root_in_states in statevertex_to_pathcondvertex[1]:
        indexes_of_roots_in_pathcond_to_pathcond = np.where(
            pathcondvertex_to_pathcondvertex[0] == _root_in_states
        )[0]
        indexes_of_roots_in_states = np.where(
            statevertex_to_pathcondvertex[1] == _root_in_states
        )[0][0]

        for index in indexes_of_roots_in_pathcond_to_pathcond:
            shift = 0
            while (
                pathcondvertex_to_pathcondvertex[0][index + 1]
                > statevertex_to_pathcondvertex[1][shift]
            ):
                shift += 1

            edge_index_pc_state.append(
                [
                    pathcondvertex_to_pathcondvertex[0][index + 1] - shift,
                    statevertex_to_pathcondvertex[0][indexes_of_roots_in_states],
                ]
            )
            edge_index_state_pc.append(
                [
                    statevertex_to_pathcondvertex[0][indexes_of_roots_in_states],
                    pathcondvertex_to_pathcondvertex[0][index + 1] - shift,
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
