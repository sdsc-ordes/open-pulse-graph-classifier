import torch
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split
import numpy as np


def split_data(data, batch_size=128):
    num_users = data["user"].num_nodes
    idx = np.arange(num_users)
    train_idx, testval_idx = train_test_split(idx, test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(testval_idx, test_size=0.5, random_state=42)

    data["user"].train_mask = torch.zeros(num_users, dtype=torch.bool)
    data["user"].val_mask = torch.zeros(num_users, dtype=torch.bool)
    data["user"].test_mask = torch.zeros(num_users, dtype=torch.bool)

    data["user"].train_mask[train_idx] = True
    data["user"].val_mask[val_idx] = True
    data["user"].test_mask[test_idx] = True

    train_loader = NeighborLoader(
        data,
        input_nodes=("user", data["user"].train_mask),
        num_neighbors=[15, 10],
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = NeighborLoader(
        data,
        input_nodes=("user", data["user"].val_mask),
        num_neighbors=[15, 10],
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = NeighborLoader(
        data,
        input_nodes=("user", data["user"].test_mask),
        num_neighbors=[15, 10],
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, val_loader
