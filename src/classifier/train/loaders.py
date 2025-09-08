import torch
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split
import numpy as np


def build_loader(data, masks, split, num_neighbors, batch_size, shuffle):
    return NeighborLoader(
        data,
        input_nodes={ntype: m[split] for ntype, m in masks.items()},
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def make_loaders(data, batch_size=128, num_neighbors=[15, 10], random_state=42):
    masks = {}

    for ntype in data.node_types:
        node_store = data[ntype]
        n = node_store.num_nodes

        # Append known anchor info as input feature (if not already added)
        if (
            hasattr(node_store, "is_anchor")
            and getattr(node_store, "_anchor_in_x", False) is False
        ):
            anchor_feat = node_store.is_anchor.view(-1, 1).float()
            node_store.x = torch.cat([node_store.x, anchor_feat], dim=1)
            node_store._anchor_in_x = True

        # Split ONLY unknown nodes into train/val/test (if they exist)
        if hasattr(node_store, "is_unknown") and node_store.is_unknown.any():
            unknown_idx = node_store.is_unknown.nonzero(as_tuple=True)[0].cpu().numpy()
            y_unknown = node_store.y[node_store.is_unknown].cpu().numpy()

            train_idx, testval_idx = train_test_split(
                unknown_idx,
                test_size=0.4,
                random_state=random_state,
                stratify=y_unknown,
            )
            y_testval = node_store.y[testval_idx].cpu().numpy()
            val_idx, test_idx = train_test_split(
                testval_idx,
                test_size=0.5,
                random_state=random_state,
                stratify=y_testval,
            )

            node_store.train_mask = torch.zeros(n, dtype=torch.bool)
            node_store.val_mask = torch.zeros(n, dtype=torch.bool)
            node_store.test_mask = torch.zeros(n, dtype=torch.bool)
            node_store.train_mask[torch.as_tensor(train_idx)] = True
            node_store.val_mask[torch.as_tensor(val_idx)] = True
            node_store.test_mask[torch.as_tensor(test_idx)] = True

            masks[ntype] = {
                "train": node_store.train_mask,
                "val": node_store.val_mask,
                "test": node_store.test_mask,
            }

    train_loader = build_loader(
        data, masks, "train", num_neighbors, batch_size, shuffle=True
    )
    val_loader = build_loader(
        data, masks, "val", num_neighbors, batch_size, shuffle=False
    )
    test_loader = build_loader(
        data, masks, "test", num_neighbors, batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


## LEGACY
# def split_data(data, batch_size=128):
#     num_users = data["user"].num_nodes
#     idx = np.arange(num_users)
#     train_idx, testval_idx = train_test_split(idx, test_size=0.4, random_state=42)
#     val_idx, test_idx = train_test_split(testval_idx, test_size=0.5, random_state=42)

#     data["user"].train_mask = torch.zeros(num_users, dtype=torch.bool)
#     data["user"].val_mask = torch.zeros(num_users, dtype=torch.bool)
#     data["user"].test_mask = torch.zeros(num_users, dtype=torch.bool)

#     data["user"].train_mask[train_idx] = True
#     data["user"].val_mask[val_idx] = True
#     data["user"].test_mask[test_idx] = True

#     train_loader = NeighborLoader(
#         data,
#         input_nodes=("user", data["user"].train_mask),
#         num_neighbors=[15, 10],
#         batch_size=batch_size,
#         shuffle=True,
#     )

#     val_loader = NeighborLoader(
#         data,
#         input_nodes=("user", data["user"].val_mask),
#         num_neighbors=[15, 10],
#         batch_size=batch_size,
#         shuffle=False,
#     )

#     test_loader = NeighborLoader(
#         data,
#         input_nodes=("user", data["user"].test_mask),
#         num_neighbors=[15, 10],
#         batch_size=batch_size,
#         shuffle=False,
#     )

#     return train_loader, test_loader, val_loader
