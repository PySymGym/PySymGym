from pathlib import Path
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
from ml.inference import TORCH
import os
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches


def read_configs(dir) -> list[Path]:
    return [
        Path(dir) / file
        for file in os.listdir(Path(dir))
        if file.endswith(".yml") or file.endswith(".yaml")
    ]


def heterodata_visualizer(heterodata: HeteroData, file_name: str):
    if not file_name.endswith(".png"):
        file_name += ".png"

    graph = to_networkx(heterodata, to_undirected=False)

    node_type_colors = {
        TORCH.game_vertex: "#2C9BE6FF",
        TORCH.state_vertex: "#ED8546",
        TORCH.path_condition_vertex: "#70B349FF",
    }

    edge_type_colors = {
        TORCH.gamevertex_to_gamevertex: "#1D73ACFF",
        TORCH.statevertex_in_gamevertex: "#B68211FF",
        TORCH.gamevertex_in_statevertex: "#B68211FF",
        TORCH.statevertex_history_gamevertex: "#37C0FFFF",
        TORCH.gamevertex_history_statevertex: "#37C0FFFF",
        TORCH.statevertex_parentof_statevertex: "#DB5C5CFF",
        TORCH.pathcondvertex_to_pathcondvertex: "#579434FF",
        TORCH.pathcondvertex_to_statevertex: "#AB51FFFF",
        TORCH.statevertex_to_pathcondvertex: "#AB51FFFF",
    }

    node_colors = []
    labels = {}
    for node, attrs in graph.nodes(data=True):
        node_type = attrs["type"]
        color = node_type_colors[node_type]
        node_colors.append(color)

        if node_type == TORCH.game_vertex:
            labels[node] = f"G{node}"
        if node_type == TORCH.state_vertex:
            labels[node] = f"S{node}"
        elif node_type == TORCH.path_condition_vertex:
            labels[node] = f"PC{node}"

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

    legend_patches = []
    for node_type, color in node_type_colors.items():
        legend_patches.append(mpatches.Patch(color=color, label=node_type))

    plt.legend(handles=legend_patches, loc="upper right")
    plt.title("Heterodata")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()
