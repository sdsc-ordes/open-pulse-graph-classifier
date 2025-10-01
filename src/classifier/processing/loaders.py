import torch
from torch_geometric.loader import NeighborLoader


def build_loader(
    data,
    ntype,
    split_idx,
    shuffle,
    batch_size=128,
    num_neighbors=[15, 10],
    random_state=42,
):
    return NeighborLoader(
        data,
        input_nodes=(ntype, torch.as_tensor(split_idx)),
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
    )
