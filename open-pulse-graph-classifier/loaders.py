import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData

from subgraphs import create_train_subgraph, create_test_subgraph, create_val_subgraph


def split_mask(num_nodes, train_ratio=0.6, val_ratio=0.2):
    """Generate train, val, and test masks."""
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    indices = torch.randperm(num_nodes)  # Shuffle indices
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train : num_train + num_val]] = True
    test_mask[indices[num_train + num_val :]] = True

    return train_mask, val_mask, test_mask


def split_data(data, batch_size=128):
    for node_type in data.node_types:
        (
            data[node_type].train_mask,
            data[node_type].val_mask,
            data[node_type].test_mask,
        ) = split_mask(data[node_type].x.shape[0])

    subgraph_train = create_train_subgraph(data)
    subgraph_test = create_test_subgraph(data)
    subgraph_val = create_val_subgraph(data)

    print("Subgraph Train:")
    print(subgraph_train)
    print("Subgraph Test:")
    print(subgraph_test)
    print("Subgraph Val:")
    print(subgraph_val)

    print("Train Loader Created")
    train_loaders = {
        node_type: NeighborLoader(
            subgraph_train,
            num_neighbors=[10, 10],
            input_nodes=(node_type, subgraph_train[node_type].train_mask),
            batch_size=batch_size,
            shuffle=True,
        )
        for node_type in data.node_types
        if subgraph_train[node_type].train_mask.nelement() > 0
    }

    print("Test Loader Created")
    test_loaders = {
        node_type: NeighborLoader(
            subgraph_test,
            num_neighbors=[10, 10],
            input_nodes=(node_type, subgraph_test[node_type].test_mask),
            batch_size=batch_size,
            shuffle=False,
        )
        for node_type in data.node_types
        if subgraph_test[node_type].test_mask.nelement() > 0
    }

    print("Validation Loader Created")
    val_loaders = {
        node_type: NeighborLoader(
            subgraph_val,
            num_neighbors=[10, 10],
            input_nodes=(node_type, subgraph_val[node_type].val_mask),
            batch_size=batch_size,
            shuffle=False,
        )
        for node_type in data.node_types
        if subgraph_val[node_type].val_mask.nelement() > 0
    }

    return train_loaders, test_loaders, val_loaders
