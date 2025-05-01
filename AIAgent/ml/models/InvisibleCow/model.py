import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm
from torch.nn.functional import log_softmax
from ml.dataset import NUM_PC_FEATURES


class StateModelEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_of_state_features,
        normalization: bool,
    ):
        super().__init__()
        self.state_conv1 = GCNConv(
            6, out_channels=hidden_channels, normalize=normalization
        )
        self.state_norm1 = BatchNorm(hidden_channels)

        self.state_history_cfg_conv1 = SAGEConv(
            (hidden_channels, 7), hidden_channels, normalize=normalization
        )
        self.state_in_cfg_conv1 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.state_to_pc_conv1 = SAGEConv(
            (hidden_channels, NUM_PC_FEATURES), hidden_channels, normalize=normalization
        )

        self.cfg_conv1 = GCNConv(
            hidden_channels, out_channels=hidden_channels, normalize=normalization
        )
        self.cfg_norm1 = BatchNorm(hidden_channels)
        self.pc_conv1 = GCNConv(
            hidden_channels, out_channels=hidden_channels, normalize=normalization
        )
        self.pc_norm1 = BatchNorm(hidden_channels)

        self.cfg_history_state_conv1 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.cfg_in_state_conv1 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.pc_to_state_conv1 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )

        self.state_conv2 = GCNConv(
            in_channels=hidden_channels, out_channels=hidden_channels
        )
        self.state_norm2 = BatchNorm(hidden_channels)

        self.state_history_cfg_conv2 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.state_in_cfg_conv2 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.state_to_pc_conv2 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )

        self.cfg_conv2 = GCNConv(
            hidden_channels, out_channels=hidden_channels, normalize=normalization
        )
        self.cfg_norm2 = BatchNorm(hidden_channels)
        self.pc_conv2 = GCNConv(
            hidden_channels, out_channels=hidden_channels, normalize=normalization
        )
        self.pc_norm2 = BatchNorm(hidden_channels)

        self.cfg_history_state_conv2 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.cfg_in_state_conv2 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.pc_to_state_conv2 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )

        self.state_conv3 = GCNConv(
            in_channels=hidden_channels, out_channels=hidden_channels
        )
        self.state_norm3 = BatchNorm(hidden_channels)

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
        state_x = self.state_conv1(state_x, edge_index_s_s).relu()
        state_x = self.state_norm1(state_x)

        game_x = self.state_history_cfg_conv1(
            (state_x, game_x), edge_index_history_s_v
        ).relu()
        game_x = self.state_in_cfg_conv1((state_x, game_x), edge_index_in_s_v).relu()
        pc_x = self.state_to_pc_conv1((state_x, pc_x), edge_index_s_pc).relu()

        game_x = self.cfg_conv1(game_x, edge_index_v_v).relu()
        game_x = self.cfg_norm1(game_x)
        pc_x = self.pc_conv1(pc_x, edge_index_pc_pc).relu()
        pc_x = self.pc_norm1(pc_x)

        state_x = self.cfg_history_state_conv1(
            (game_x, state_x), edge_index_history_v_s
        ).relu()
        state_x = self.cfg_in_state_conv1((game_x, state_x), edge_index_in_v_s).relu()
        state_x = self.pc_to_state_conv1((pc_x, state_x), edge_index_pc_s).relu()

        state_x = self.state_conv2(state_x, edge_index_s_s).relu()
        state_x = self.state_norm2(state_x)

        game_x = self.state_history_cfg_conv2(
            (state_x, game_x), edge_index_history_s_v
        ).relu()
        game_x = self.state_in_cfg_conv2((state_x, game_x), edge_index_in_s_v).relu()
        pc_x = self.state_to_pc_conv2((state_x, pc_x), edge_index_s_pc).relu()

        game_x = self.cfg_conv2(game_x, edge_index_v_v).relu()
        game_x = self.cfg_norm2(game_x)
        pc_x = self.pc_conv2(pc_x, edge_index_pc_pc)
        pc_x = self.pc_norm2(pc_x)

        state_x = self.cfg_history_state_conv2(
            (game_x, state_x), edge_index_history_v_s
        ).relu()
        state_x = self.cfg_in_state_conv2((game_x, state_x), edge_index_in_v_s).relu()
        state_x = self.pc_to_state_conv2((pc_x, state_x), edge_index_pc_s).relu()

        state_x = self.state_conv3(state_x, edge_index_s_s).relu()
        state_x = self.state_norm3(state_x)

        state_x = self.lin(state_x).relu()
        return log_softmax(self.lin_last(state_x), dim=0)
