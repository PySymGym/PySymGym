import torch


def min_max_scaling(game_x, state_x, game_x_batch, y_true_batch, eps=1e-12):
    scaled_game_x = torch.clone(game_x)
    scaled_state_x = torch.clone(state_x)
    batch_numbers = torch.unique(y_true_batch)
    for elem_number in batch_numbers:
        scaled_game_x[:, 1][game_x_batch == elem_number] = (
            game_x[:, 1][game_x_batch == elem_number]
            - torch.min(game_x[:, 1][game_x_batch == elem_number])
        ) / (
            torch.max(game_x[:, 1][game_x_batch == elem_number])
            - torch.min(game_x[:, 1][game_x_batch == elem_number])
            + eps
        )
        scaled_state_x[y_true_batch == elem_number] = (
            state_x[y_true_batch == elem_number]
            - torch.min(state_x[y_true_batch == elem_number], dim=0).values
        ) / (
            torch.max(state_x[y_true_batch == elem_number], dim=0).values
            - torch.min(state_x[y_true_batch == elem_number], dim=0).values
            + eps
        )
    return scaled_game_x, scaled_state_x
