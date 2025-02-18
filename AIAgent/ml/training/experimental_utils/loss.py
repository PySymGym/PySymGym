import torch


def euclidean_dist(y_pred, y_true):
    if len(y_pred) > 1:
        y_pred_min, _ = torch.min(y_pred, dim=0)
        y_pred_norm = y_pred - y_pred_min

        y_true_min, _ = torch.min(y_true, dim=0)
        y_true_norm = y_true - y_true_min
        return torch.sqrt(torch.sum((y_pred_norm - y_true_norm) ** 2))
    else:
        return 0
