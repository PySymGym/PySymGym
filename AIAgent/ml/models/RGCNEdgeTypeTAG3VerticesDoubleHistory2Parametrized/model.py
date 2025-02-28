import torch
from torch.nn import Linear
from torch_geometric.nn import (
    TAGConv,
    SAGEConv,
    RGCNConv,
    ResGatedGraphConv,
    HeteroConv,
)
from torch.nn.functional import log_softmax

from ml.inference import TORCH
import copy


class StateModelEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_of_state_features,
        num_hops_1,
        num_hops_2,
        normalization: bool,
    ):
        super().__init__()
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, 3)
        self.conv10 = TAGConv(7, hidden_channels, num_hops_1, normalize=normalization)
        self.conv2 = TAGConv(
            hidden_channels, hidden_channels, num_hops_2, normalize=normalization
        )
        self.conv3 = ResGatedGraphConv(
            (hidden_channels, 7), hidden_channels, edge_dim=2
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
        self.lin = Linear(hidden_channels, num_of_state_features)
        self.lin_last = Linear(num_of_state_features, 1)

    def forward(
        self,
        game_x,
        state_x,
        edge_index_v_v,
        edge_type_v_v,
        edge_index_history_v_s,
        edge_attr_history_v_s,
        edge_index_in_v_s,
        edge_index_s_s,
    ):
        game_x = self.conv10(game_x, edge_index_v_v).relu()

        if edge_type_v_v.numel() != 0:
            game_x = self.conv1(game_x, edge_index_v_v, edge_type_v_v).relu()

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


class HeteroStateModelEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_of_state_features,
        num_hops_1,
        num_hops_2,
        normalization: bool,
    ):
        super().__init__()
        self.hetero_conv1 = HeteroConv(
            {
                TORCH.gamevertex_to_gamevertex: TAGConv(
                    7,
                    hidden_channels,
                    num_hops_1,
                    normalize=normalization,
                ),
            }
        )
        self.hetero_conv2 = HeteroConv(
            {
                TORCH.gamevertex_to_gamevertex: RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    3,
                ),
            }
        )
        self.hetero_conv20 = HeteroConv(
            {
                TORCH.gamevertex_history_statevertex: ResGatedGraphConv(
                    (hidden_channels, 7),
                    hidden_channels,
                    edge_dim=2,
                ),
            }
        )
        self.hetero_conv3 = HeteroConv(
            {
                TORCH.gamevertex_history_statevertex: SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    normalize=normalization,
                ),
            }
        )
        self.hetero_conv4 = HeteroConv(
            {
                TORCH.gamevertex_in_statevertex: SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    normalize=normalization,
                )
            }
        )
        self.hetero_conv5 = HeteroConv(
            {
                TORCH.gamevertex_in_statevertex: SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    normalize=normalization,
                )
            }
        )
        self.hetero_conv6 = HeteroConv(
            {
                TORCH.statevertex_parentof_statevertex: TAGConv(
                    hidden_channels,
                    hidden_channels,
                    num_hops_2,
                    normalize=normalization,
                )
            }
        )
        self.hetero_conv7 = HeteroConv(
            {
                TORCH.statevertex_parentof_statevertex: SAGEConv(
                    hidden_channels,
                    hidden_channels,
                    normalize=normalization,
                )
            }
        )
        self.lin = Linear(hidden_channels, num_of_state_features)
        self.lin_last = Linear(num_of_state_features, 1)

    def forward(self, x_dict, edge_index_dict, edge_type_dict, edge_attr_dict):
        x_dict_copy = dict(x_dict)
        game_x_dict = self.hetero_conv1(x_dict_copy, edge_index_dict)
        game_x_dict = self.hetero_conv2(
            game_x_dict,
            edge_index_dict,
            edge_type_dict,
        )
        x_dict_copy[TORCH.game_vertex] = game_x_dict[TORCH.game_vertex]
        state_x_dict = self.hetero_conv20(x_dict_copy, edge_index_dict, edge_attr_dict)
        x_dict_copy[TORCH.state_vertex] = state_x_dict[TORCH.state_vertex]
        state_x_dict = self.hetero_conv3(x_dict_copy, edge_index_dict)
        x_dict_copy[TORCH.state_vertex] = state_x_dict[TORCH.state_vertex]
        state_x_dict = self.hetero_conv4(x_dict_copy, edge_index_dict)
        x_dict_copy[TORCH.state_vertex] = state_x_dict[TORCH.state_vertex]
        state_x_dict = self.hetero_conv5(x_dict_copy, edge_index_dict)
        x_dict_copy[TORCH.state_vertex] = state_x_dict[TORCH.state_vertex]
        state_x_dict = self.hetero_conv6(x_dict_copy, edge_index_dict)
        x_dict_copy[TORCH.state_vertex] = state_x_dict[TORCH.state_vertex]
        state_x_dict = self.hetero_conv7(x_dict_copy, edge_index_dict)
        state_x = self.lin(state_x_dict[TORCH.state_vertex]).relu()
        return log_softmax(self.lin_last(state_x), dim=0)
