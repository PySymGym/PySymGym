import torch
from torch.nn import Linear, ReLU, ModuleList
from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm, Sequential
from torch.nn.functional import log_softmax
from ml.dataset import NUM_PC_FEATURES, NUM_CFG_VERTEX_FEATURES, NUM_STATE_FEATURES


class Block(torch.nn.Module):
    def __init__(
        self,
        pc_channels,
        cfg_vertex_channels,
        hidden_channels,
        normalization,
    ):
        super().__init__()
        self.state_history_cfg_conv = SAGEConv(
            (hidden_channels, cfg_vertex_channels),
            hidden_channels,
            normalize=normalization,
        )
        self.state_in_cfg_conv = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.state_to_pc_conv = SAGEConv(
            (hidden_channels, pc_channels), hidden_channels, normalize=normalization
        )

        self.cfg_conv = Sequential(
            "x, edge_index",
            [
                (
                    GCNConv(
                        hidden_channels,
                        out_channels=hidden_channels,
                        normalize=normalization,
                    ),
                    "x, edge_index -> x",
                ),
                (ReLU(), "x -> x"),
                (BatchNorm(hidden_channels), "x -> x"),
            ],
        )
        self.pc_conv = Sequential(
            "x, edge_index",
            [
                (
                    GCNConv(
                        hidden_channels,
                        out_channels=hidden_channels,
                        normalize=normalization,
                    ),
                    "x, edge_index -> x",
                ),
                (ReLU(), "x -> x"),
                (BatchNorm(hidden_channels), "x -> x"),
            ],
        )

        self.cfg_history_state_conv = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.cfg_in_state_conv = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.pc_to_state_conv = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )

        self.state_conv = Sequential(
            "x, edge_index",
            [
                (
                    GCNConv(in_channels=hidden_channels, out_channels=hidden_channels),
                    "x, edge_index -> x",
                ),
                (ReLU(), "x -> x"),
                (BatchNorm(hidden_channels), "x -> x"),
            ],
        )

    def forward(
        self,
        game_x,
        state_x,
        pc_x,
        edge_index_v_v,
        edge_index_history_v_s,
        edge_index_history_s_v,
        edge_index_in_v_s,
        edge_index_in_s_v,
        edge_index_s_s,
        edge_index_pc_pc,
        edge_index_pc_s,
        edge_index_s_pc,
    ):
        game_x = self.state_history_cfg_conv(
            (state_x, game_x), edge_index_history_s_v
        ).relu()
        game_x = self.state_in_cfg_conv((state_x, game_x), edge_index_in_s_v).relu()
        pc_x = self.state_to_pc_conv((state_x, pc_x), edge_index_s_pc).relu()

        game_x = self.cfg_conv(game_x, edge_index_v_v)
        pc_x = self.pc_conv(pc_x, edge_index_pc_pc)

        state_x = self.cfg_history_state_conv(
            (game_x, state_x), edge_index_history_v_s
        ).relu()
        state_x = self.cfg_in_state_conv((game_x, state_x), edge_index_in_v_s).relu()
        state_x = self.pc_to_state_conv((pc_x, state_x), edge_index_pc_s).relu()

        state_x = self.state_conv(state_x, edge_index_s_s)
        return state_x, game_x, pc_x


class StateModelEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_of_state_features,
        num_blocks,
        normalization: bool,
    ):
        super().__init__()

        self.blocks = ModuleList()

        self.state_conv0 = Sequential(
            "x, edge_index",
            [
                (
                    GCNConv(
                        NUM_STATE_FEATURES,
                        out_channels=hidden_channels,
                        normalize=normalization,
                    ),
                    "x, edge_index -> x",
                ),
                (ReLU(inplace=True), "x -> x"),
                (BatchNorm(hidden_channels), "x -> x"),
            ],
        )

        self.blocks.append(
            Block(
                NUM_PC_FEATURES,
                NUM_CFG_VERTEX_FEATURES,
                hidden_channels,
                normalization,
            )
        )

        for _ in range(num_blocks - 1):
            self.blocks.append(
                Block(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels,
                    normalization,
                )
            )

        self.lin = Linear(hidden_channels, num_of_state_features)
        self.lin_last = Linear(num_of_state_features, 1)

    def forward(
        self,
        game_x,
        state_x,
        pc_x,
        edge_index_v_v,
        edge_type_v_v,
        edge_index_history_v_s,
        edge_index_history_s_v,
        edge_attr_history_v_s,
        edge_index_in_v_s,
        edge_index_in_s_v,
        edge_index_s_s,
        edge_index_pc_pc,
        edge_index_pc_s,
        edge_index_s_pc,
    ):
        state_x = self.state_conv0(state_x, edge_index_s_s)

        for block in self.blocks:
            state_x, game_x, pc_x = block(
                game_x,
                state_x,
                pc_x,
                edge_index_v_v,
                edge_index_history_v_s,
                edge_index_history_s_v,
                edge_index_in_v_s,
                edge_index_in_s_v,
                edge_index_s_s,
                edge_index_pc_pc,
                edge_index_pc_s,
                edge_index_s_pc,
            )
        state_x = self.lin(state_x).relu()
        return log_softmax(self.lin_last(state_x), dim=0)
