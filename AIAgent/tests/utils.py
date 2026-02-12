from pathlib import Path
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
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


def heterodata_vesualizer(heterodata: HeteroData):
    graph = to_networkx(heterodata, to_undirected=False)

    node_type_colors = {
        "game_vertex": "#2C9BE6FF",
        "state_vertex": "#ED8546",
        "path_condition_vertex": "#70B349FF",
    }

    edge_type_colors = {
        ("game_vertex", "to", "game_vertex"): "#1D73ACFF",
        ("state_vertex", "in", "game_vertex"): "#B68211FF",
        ("game_vertex", "in", "state_vertex"): "#B68211FF",
        ("state_vertex", "history", "game_vertex"): "#37C0FFFF",
        ("game_vertex", "history", "state_vertex"): "#37C0FFFF",
        ("state_vertex", "parent_of", "state_vertex"): "#DB5C5CFF",
        ("path_condition_vertex", "to", "path_condition_vertex"): "#579434FF",
        ("path_condition_vertex", "to", "state_vertex"): "#AB51FFFF",
        ("state_vertex", "to", "path_condition_vertex"): "#AB51FFFF",
    }

    node_colors = []
    labels = {}
    for node, attrs in graph.nodes(data=True):
        node_type = attrs["type"]
        color = node_type_colors[node_type]
        node_colors.append(color)

        if node_type == "game_vertex":
            labels[node] = f"G{node}"
        if node_type == "state_vertex":
            labels[node] = f"S{node}"
        elif node_type == "path_condition_vertex":
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
    plt.savefig("graph_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Граф сохранен в 'graph_visualization.png'")
