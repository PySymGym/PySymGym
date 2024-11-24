import torch


# def min_max_scaling(data):
#     scaled_data = data.clone()
#     scaled_data[TORCH.state_vertex].x = (
#         data[TORCH.state_vertex].x - torch.min(data[TORCH.state_vertex].x, dim=0).values
#     ) / (
#         torch.max(data[TORCH.state_vertex].x, dim=0).values
#         - torch.min(data[TORCH.state_vertex].x, dim=0).values
#         + 1e-12
#     )
#     scaled_data[TORCH.game_vertex].x[:, 1] = (
#         data[TORCH.game_vertex].x[:, 1] - torch.min(data[TORCH.game_vertex].x[:, 1])
#     ) / (
#         torch.max(data[TORCH.game_vertex].x[:, 1])
#         - torch.min(data[TORCH.game_vertex].x[:, 1])
#         + 1e-12
#     )
#     return scaled_data


def min_max_scaling(game_x, state_x, eps=1e-12):
    scaled_game_x = torch.clone(game_x)
    scaled_game_x[:, 1] = (game_x[:, 1] - torch.min(game_x[:, 1])) / (
        torch.max(game_x[:, 1]) - torch.min(game_x[:, 1]) + eps
    )
    scaled_state_x = (state_x - torch.min(state_x, dim=0).values) / (
        torch.max(state_x, dim=0).values - torch.min(state_x, dim=0).values + eps
    )
    return scaled_game_x, scaled_state_x