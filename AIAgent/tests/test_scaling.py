from ml.training.scaling import min_max_scaling
import torch


class TestMinMaxScaling:
    def test_with_multiple_elements_in_batch(self):
        game_x = torch.tensor(
            [
                [1, 2, 1, 0, 1, 1],
                [1, 5, 1, 0, 1, 1],
                [1, 3, 1, 0, 1, 1],
                [1, 45, 1, 0, 1, 1],
                [1, 89, 1, 0, 1, 1],
                [1, 4, 1, 0, 1, 1],
                [1, 6, 1, 0, 1, 1],
                [1, 2, 1, 0, 1, 1],
                [1, 2, 1, 0, 1, 1],
            ],
            dtype=torch.float,
        )
        state_x = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7],
                [1, 26, 13, 54, 35, 62, 7],
                [1, 2, 0, 4, 8, 6, 9],
                [3, 22, 3, 6, 5, 8, 7],
                [1, 2, 3, 4, 5, 6, 7],
                [5, 2, 3, 4, 5, 6, 7],
                [11, 5, 3, 47, 5, 64, 7],
            ],
            dtype=torch.float,
        )
        game_x_batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
        y_true_batch = torch.tensor([0, 1, 1, 1, 2, 2, 2])
        answer_scaled_game_x = torch.tensor(
            [
                [1, 0, 1, 0, 1, 1],
                [1, 3 / (3 + 1e-12), 1, 0, 1, 1],
                [1, 0, 1, 0, 1, 1],
                [1, 42 / (86 + 1e-12), 1, 0, 1, 1],
                [1, 86 / (86 + 1e-12), 1, 0, 1, 1],
                [1, 2 / (4 + 1e-12), 1, 0, 1, 1],
                [1, 4 / (4 + 1e-12), 1, 0, 1, 1],
                [1, 0, 1, 0, 1, 1],
                [1, 0, 1, 0, 1, 1],
            ]
        )
        answer_scaled_state_x = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [
                    0,
                    24 / (24 + 1e-12),
                    13 / (13 + 1e-12),
                    50 / (50 + 1e-12),
                    30 / (30 + 1e-12),
                    56 / (56 + 1e-12),
                    0,
                ],
                [0, 0, 0, 0, 3 / (30 + 1e-12), 0, 2 / (2 + 1e-12)],
                [
                    2 / (2 + 1e-12),
                    20 / (24 + 1e-12),
                    3 / (13 + 1e-12),
                    2 / (50 + 1e-12),
                    0,
                    2 / (56 + 1e-12),
                    0,
                ],
                [0, 0, 0, 0, 0, 0, 0],
                [4 / (10 + 1e-12), 0, 0, 0, 0, 0, 0],
                [
                    10 / (10 + 1e-12),
                    3 / (3 + 1e-12),
                    0,
                    43 / (43 + 1e-12),
                    0,
                    58 / (58 + 1e-12),
                    0,
                ],
            ]
        )
        scaled_game_x, scaled_state_x = min_max_scaling(
            game_x, state_x, game_x_batch, y_true_batch
        )
        assert torch.all(torch.isclose(answer_scaled_game_x, scaled_game_x))
        assert torch.all(torch.isclose(answer_scaled_state_x, scaled_state_x))
