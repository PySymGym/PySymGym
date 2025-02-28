import torch
from ml.inference import TORCH


def state_first_forward(encoder: torch.nn.Module):
    orig_forward = encoder.forward

    def new_forward(
        self,
        state_x,
        edge_index_s_s,
        game_x,
        edge_index_v_v,
        edge_type_v_v,
        edge_index_history_v_s,
        edge_attr_history_v_s,
        edge_index_in_v_s,
    ):
        return orig_forward(
            self,
            game_x=game_x,
            state_x=state_x,
            edge_index_v_v=edge_index_v_v,
            edge_type_v_v=edge_type_v_v,
            edge_index_history_v_s=edge_index_history_v_s,
            edge_attr_history_v_s=edge_attr_history_v_s,
            edge_index_in_v_s=edge_index_in_v_s,
            edge_index_s_s=edge_index_s_s,
        )

    encoder.forward = new_forward
    return encoder


def cfg_first_forward(encoder: torch.nn.Module):
    orig_forward = encoder.forward

    def new_forward(
        self,
        game_x,
        edge_index_v_v,
        state_x,
        edge_index_s_s,
        edge_type_v_v,
        edge_index_history_v_s,
        edge_attr_history_v_s,
        edge_index_in_v_s,
    ):
        return orig_forward(
            self,
            game_x=game_x,
            state_x=state_x,
            edge_index_v_v=edge_index_v_v,
            edge_type_v_v=edge_type_v_v,
            edge_index_history_v_s=edge_index_history_v_s,
            edge_attr_history_v_s=edge_attr_history_v_s,
            edge_index_in_v_s=edge_index_in_v_s,
            edge_index_s_s=edge_index_s_s,
        )

    encoder.forward = new_forward
    return encoder


def hetero_forward(encoder: torch.nn.Module):
    orig_forward = encoder.forward

    def new_forward(
        self,
        x_dict,
        edge_index_dict,
        edge_type_dict,
        edge_attr_dict,
    ):
        return orig_forward(
            self,
            game_x=x_dict[TORCH.game_vertex],
            state_x=x_dict[TORCH.state_vertex],
            edge_index_v_v=edge_index_dict[TORCH.gamevertex_to_gamevertex],
            edge_type_v_v=edge_type_dict[TORCH.gamevertex_to_gamevertex],
            edge_index_history_v_s=edge_index_dict[
                TORCH.gamevertex_history_statevertex
            ],
            edge_attr_history_v_s=edge_attr_dict[TORCH.gamevertex_history_statevertex],
            edge_index_in_v_s=edge_index_dict[TORCH.gamevertex_in_statevertex],
            edge_index_s_s=edge_index_dict[TORCH.statevertex_parentof_statevertex],
        )

    encoder.forward = new_forward
    return encoder
