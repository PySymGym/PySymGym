import torch
from networkx.algorithms.dominance import immediate_dominators
from networkx.classes import DiGraph
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils.convert import to_networkx

from ml.inference import TORCH


def remove_call_return_edges(cfg: Data) -> Data:
    shift = 0
    call_return_edges_types = [1, 2]
    for edge_idx, edge_type in enumerate(cfg.edge_attr):
        if edge_type in call_return_edges_types:
            cfg.edge_index = torch.cat(
                (
                    cfg.edge_index[:, 0 : edge_idx - shift],
                    cfg.edge_index[:, edge_idx + 1 - shift :],
                ),
                dim=1,
            )
            shift += 1
    return cfg


def find_entry_points(cfg: Data) -> list[int]:
    entry_points = []
    for vertex_number in range(cfg.x.size()[0]):
        if vertex_number not in cfg.edge_index[1]:
            entry_points.append(vertex_number)
    return entry_points


def find_dominators_in_cfg(graph: HeteroData) -> HeteroData:
    cfg = Data(
        graph[TORCH.game_vertex].x,
        graph[*TORCH.gamevertex_to_gamevertex].edge_index,
        graph[*TORCH.gamevertex_to_gamevertex].edge_type,
    )
    cfg = remove_call_return_edges(cfg)

    entry_points = find_entry_points(cfg)
    if not entry_points:
        raise ValueError("There is no entry points to find dominators from.")
    dominators_graphs: list[DiGraph] = list()
    networkx_cfg: DiGraph = to_networkx(cfg)
    for start in entry_points:
        dominators_graphs.append(
            DiGraph(immediate_dominators(networkx_cfg, start).items())
        )

    for dominators_graph in dominators_graphs:
        for edge in dominators_graph.edges:
            if edge[0] != edge[1]:
                graph[*TORCH.gamevertex_to_gamevertex].edge_index = torch.cat(
                    (
                        graph[*TORCH.gamevertex_to_gamevertex].edge_index,
                        torch.tensor([[edge[1]], [edge[0]]]),
                    ),
                    dim=1,
                )
                graph[*TORCH.gamevertex_to_gamevertex].edge_type = torch.cat(
                    (
                        graph[*TORCH.gamevertex_to_gamevertex].edge_type,
                        torch.tensor([3]),
                    )
                )
