from ml.training.utils import possibility_loss
import torch


class TestPossibilityLoss:
    def test_one_element_in_batch(self):
        y_pred = torch.tensor([[0.6], [0.4]])
        y_true = torch.tensor([[1.0], [0.0]])
        assert torch.isclose(
            possibility_loss(y_pred, y_true, torch.tensor([0, 0])), torch.tensor(-0.2)
        )
        y_pred = torch.tensor([[0.4], [0.6]])
        y_true = torch.tensor([[1.0], [0.0]])
        assert torch.isclose(
            possibility_loss(y_pred, y_true, torch.tensor([0, 0])), torch.tensor(0.2)
        )
        y_pred = torch.tensor([[0.33], [0.33], [0.33]])
        y_true = torch.tensor([[0.5], [0.0], [0.5]])
        assert torch.isclose(
            possibility_loss(y_pred, y_true, torch.tensor([0, 0, 0])), torch.tensor(0.0)
        )
        y_pred = torch.tensor([[0.4], [0.3], [0.3]])
        y_true = torch.tensor([[0.5], [0.0], [0.5]])
        assert torch.isclose(
            possibility_loss(y_pred, y_true, torch.tensor([0, 0, 0])),
            torch.tensor(-0.1),
        )
        y_pred = torch.tensor([[0.4], [0.3], [0.3]])
        y_true = torch.tensor([[0.33], [0.33], [0.33]])
        assert torch.isclose(
            possibility_loss(y_pred, y_true, torch.tensor([0, 0, 0])),
            torch.tensor(-0.4),
        )

    def test_multiple_elements_in_batch(self):
        y_pred = torch.tensor([[0.6], [0.4]]).repeat(4, 1)
        y_true = torch.tensor([[1.0], [0.0]]).repeat(4, 1)
        assert torch.isclose(
            possibility_loss(y_pred, y_true, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])),
            torch.tensor(-0.2),
        )
        y_pred = torch.tensor(
            [[0.6], [0.4], [0.33], [0.33], [0.33], [0.4], [0.3], [0.3]]
        )
        y_true = torch.tensor([[1.0], [0.0], [0.5], [0.0], [0.5], [0.5], [0.0], [0.5]])
        assert torch.isclose(
            possibility_loss(y_pred, y_true, torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])),
            torch.tensor(-0.1),
        )
