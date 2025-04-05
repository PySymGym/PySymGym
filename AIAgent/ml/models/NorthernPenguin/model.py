import torch
from torch.nn import Linear
from torch_geometric.nn import TAGConv, SAGEConv, RGCNConv, ResGatedGraphConv, GCN
from torch.nn.functional import log_softmax
from ml.dataset import NUM_PC_FEATURES


class StateModelEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_of_state_features,
        num_hops_1,
        num_hops_2,
        normalization: bool,
        num_pc_layers,
    ):
        super().__init__()
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, 3)
        self.conv10 = TAGConv(7, hidden_channels, num_hops_1, normalize=normalization)
        self.conv2 = TAGConv(
            hidden_channels, hidden_channels, num_hops_2, normalize=normalization
        )
        self.conv3 = ResGatedGraphConv(
            (hidden_channels, 6), hidden_channels, edge_dim=2
        )
        self.conv32 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.conv4 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.conv42 = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.conv5 = SAGEConv(hidden_channels, hidden_channels, normalize=normalization)
        self.pc_conv = GCN(
            NUM_PC_FEATURES,
            hidden_channels,
            num_layers=num_pc_layers,
            out_channels=hidden_channels,
        )
        self.pc_to_state = SAGEConv(
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
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
        edge_attr_history_v_s,
        edge_index_in_v_s,
        edge_index_s_s,
        edge_index_pc_pc,
        edge_index_pc_state,
        edge_index_state_pc,
    ):
        game_x = self.conv10(game_x, edge_index_v_v).relu()

        if edge_type_v_v.numel() != 0:
            game_x = self.conv1(game_x, edge_index_v_v, edge_type_v_v).relu()
        pc_x = self.pc_conv(pc_x, edge_index_pc_pc).relu()
        state_x = self.conv3(
            (game_x, state_x),
            edge_index_history_v_s,
            edge_attr_history_v_s,
        ).relu()

        state_x = self.conv32(
            (game_x, state_x),
            edge_index_history_v_s,
        ).relu()

        state_x = self.conv4(
            (game_x, state_x),
            edge_index_in_v_s,
        ).relu()

        state_x = self.conv42(
            (game_x, state_x),
            edge_index_in_v_s,
        ).relu()

        state_x = self.pc_to_state((pc_x, state_x), edge_index_pc_state).relu()
        state_x = self.conv2(
            state_x,
            edge_index_s_s,
        ).relu()

        state_x = self.conv5(
            state_x,
            edge_index_s_s,
        ).relu()
        state_x = self.lin(state_x).relu()
        return log_softmax(self.lin_last(state_x), dim=0)
