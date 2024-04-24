import os.path
import pickle
from random import shuffle

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import tqdm
from config import GeneralConfig

BALANCE_DATASET = False


def get_module_name(clazz):
    return clazz.__module__.split(".")[-2]


class HetGNNTestTrain:
    """predicts ExpectedStateNumber using Heterogeneous GNN"""

    def __init__(self, model_class, hidden):
        self.state_maps = {}
        self.model_class = model_class
        self.hidden = hidden

    def train_and_save(self, dataset_dir, epochs, dir_to_save):
        dataset = []
        for file in os.listdir(dataset_dir):
            print(file)
            with open(os.path.join(dataset_dir, file), "rb") as f:
                dat = pickle.load(f)
                if BALANCE_DATASET and len(dat) > 5000:
                    dat = dat[:5000]
                    print("Part of the dataset is chosen!")
                dataset.extend(dat)

        torch.manual_seed(12345)
        shuffle(dataset)

        split_at = round(len(dataset) * 0.85)

        train_dataset = dataset[:split_at]
        test_dataset = dataset[split_at:]

        print(f"Number of training graphs: {len(train_dataset)}")
        print(f"Number of test graphs: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=100, shuffle=False
        )  # TODO: add learning by batches!
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = self.model_class(hidden_channels=self.hidden, out_channels=8).to(
            GeneralConfig.DEVICE
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        for epoch in range(1, epochs + 1):
            self.train(model, train_loader, optimizer, GeneralConfig.DEVICE)
            train_acc = self.tst(model, train_loader, GeneralConfig.DEVICE)
            test_acc = self.tst(model, test_loader, GeneralConfig.DEVICE)
            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_acc:.6f}, Test Loss: {test_acc:.6f}"
            )

        self.save_simple(model, dir_to_save, epochs)

    # loss function from link prediction example
    def weighted_mse_loss(self, pred, target, weight=None):
        weight = 1.0 if weight is None else weight[target].to(pred.dtype)
        return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

    def train(self, model, train_loader, optimizer, device):
        model.train()
        for data in tqdm.tqdm(
            train_loader, desc="Pretraining"
        ):  # Iterate in batches over the training dataset.
            data = data.to(device)
            optimizer.zero_grad()  # Clear gradients.
            out = model(
                data.x_dict["game_vertex"],
                data.x_dict["state_vertex"],
                data.edge_index_dict["game_vertex", "to", "game_vertex"],
                data["game_vertex", "to", "game_vertex"].edge_type,
                data["game_vertex", "history", "state_vertex"].edge_index,
                data["game_vertex", "history", "state_vertex"].edge_attr,
                data["game_vertex", "in", "state_vertex"].edge_index,
                data["state_vertex", "parent_of", "state_vertex"].edge_index,
            )
            target = data.y
            loss = F.mse_loss(out, target)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

    @torch.no_grad()
    def tst(self, model, loader, device):
        model.eval()
        total_loss = 0
        number_of_states_total = 0
        for data in tqdm.tqdm(loader, desc="Test"):
            data.to(device)
            out = model(
                data.x_dict["game_vertex"],
                data.x_dict["state_vertex"],
                data["game_vertex", "to", "game_vertex"].edge_index,
                data["game_vertex", "to", "game_vertex"].edge_type,
                data["game_vertex", "history", "state_vertex"].edge_index,
                data["game_vertex", "history", "state_vertex"].edge_attr,
                data["game_vertex", "in", "state_vertex"].edge_index,
                data["state_vertex", "parent_of", "state_vertex"].edge_index,
            )
            target = data.y
            for i, x in enumerate(out):
                loss = F.mse_loss(x, target[i])
                total_loss += loss
                number_of_states_total += 1
        return total_loss / number_of_states_total  # correct / len(loader.dataset)

    def save_simple(self, model, dir, epochs):
        dir = os.path.join(
            dir,
            get_module_name(self.model_class),
            str(self.hidden) + "ch",
            str(epochs) + "e",
        )
        if not os.path.exists(dir):
            os.makedirs(dir)
        filepath = os.path.join(dir, "GNN_state_pred_het_dict")
        # case 1
        torch.save(model.state_dict(), filepath)
        # case 2
        filepath = os.path.join(dir, "GNN_state_pred_het_full")
        torch.save(model, filepath)
