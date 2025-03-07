import torch
from torch_geometric.data import HeteroData
import numpy as np


def extract_edges(edge_indices, src_mask, dst_mask):
    # Find edges where both nodes are in the set
    src = edge_indices[0]
    dst = edge_indices[1]
    src_mask = src_mask.to(torch.bool)
    dst_mask = dst_mask.to(torch.bool)

    if (True in src_mask) and (True in dst_mask):
        edges_masked = []
        for edge in edge_indices:
            if edge[0] in src[src_mask] and edge[1] in dst[dst_mask]:
                edges_masked.append(edge)
        edges_masked = torch.Tensor(edges_masked)
        if edges_masked.nelement() != 0:
            return edges_masked
        else:
            # no edges after the masking
            return None
    else:
        # no edges even before the masking
        return None


def prep_edges_mask(edges, nodes_mask, nodes):
    edge_mask = []
    nodes_array = nodes.numpy()
    for edge in edges:
        index_data = np.where(nodes_array == edge)
        mask_value = nodes_mask[index_data].item()
        edge_mask.append(mask_value)
    return torch.Tensor(edge_mask)


def create_train_subgraph(data):
    """Creates a subgraph containing only training nodes and valid edges."""
    subgraph = HeteroData()

    for node_type in data.node_types:
        mask = data[node_type].train_mask
        masked_data = mask * data[node_type].x
        subgraph[node_type].x = masked_data[masked_data != 0]
        subgraph[node_type].train_mask = mask[mask]
        # set num_nodes for Neighborloader later
        subgraph[node_type].num_nodes = subgraph[node_type].x.shape[0]

    # Filter edges: Keep only edges where both nodes are in the set
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_indices = data[edge_type].edge_index
        edges_srcs = edge_indices[0]
        node_src_mask = data[src_type].train_mask
        node_src = data[src_type].x
        edges_dsts = edge_indices[1]
        node_dst_mask = data[dst_type].train_mask
        node_dst = data[dst_type].x
        edges_srcs_mask = prep_edges_mask(edges_srcs, node_src_mask, node_src)
        edges_dsts_mask = prep_edges_mask(edges_dsts, node_dst_mask, node_dst)

        edges_masked = extract_edges(edge_indices, edges_srcs_mask, edges_dsts_mask)
        if edges_masked:
            subgraph[edge_type].edge_index = edges_masked
    return subgraph


def create_test_subgraph(data):
    """Creates a subgraph containing only testing nodes and valid edges."""
    subgraph = HeteroData()

    for node_type in data.node_types:
        mask = data[node_type].test_mask
        masked_data = mask * data[node_type].x
        subgraph[node_type].x = masked_data[masked_data != 0]
        subgraph[node_type].test_mask = mask[mask]
        # set num_nodes for Neighborloader later
        subgraph[node_type].num_nodes = subgraph[node_type].x.shape[0]

    # Filter edges: Keep only edges where both nodes are in the set
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_indices = data[edge_type].edge_index
        edges_srcs = edge_indices[0]
        node_src_mask = data[src_type].test_mask
        node_src = data[src_type].x
        edges_dsts = edge_indices[1]
        node_dst_mask = data[dst_type].test_mask
        node_dst = data[dst_type].x
        edges_srcs_mask = prep_edges_mask(edges_srcs, node_src_mask, node_src)
        edges_dsts_mask = prep_edges_mask(edges_dsts, node_dst_mask, node_dst)

        edges_masked = extract_edges(edge_indices, edges_srcs_mask, edges_dsts_mask)
        if edges_masked:
            subgraph[edge_type].edge_index = edges_masked
    return subgraph


def create_val_subgraph(data):
    """Creates a subgraph containing only valing nodes and valid edges."""
    subgraph = HeteroData()

    for node_type in data.node_types:
        mask = data[node_type].val_mask
        masked_data = mask * data[node_type].x
        subgraph[node_type].x = masked_data[masked_data != 0]
        subgraph[node_type].val_mask = mask[mask]
        # set num_nodes for Neighborloader later
        subgraph[node_type].num_nodes = subgraph[node_type].x.shape[0]

    # Filter edges: Keep only edges where both nodes are in the set
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_indices = data[edge_type].edge_index
        edges_srcs = edge_indices[0]
        node_src_mask = data[src_type].val_mask
        node_src = data[src_type].x
        edges_dsts = edge_indices[1]
        node_dst_mask = data[dst_type].val_mask
        node_dst = data[dst_type].x
        edges_srcs_mask = prep_edges_mask(edges_srcs, node_src_mask, node_src)
        edges_dsts_mask = prep_edges_mask(edges_dsts, node_dst_mask, node_dst)

        edges_masked = extract_edges(edge_indices, edges_srcs_mask, edges_dsts_mask)
        if edges_masked:
            subgraph[edge_type].edge_index = edges_masked

    return subgraph
