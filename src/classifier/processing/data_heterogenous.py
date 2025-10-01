import torch
from torch_geometric.data import HeteroData

from classifier.processing.data_transformer import (
    identify_anchors,
    global_local_matcher,
    save_index_mapping,
    data_transformer,
    vectorize_features,
)


def create_heterogenous_data(
    nodes_ids,
    nodes_features,
    edges_indices,
    relationships,
    train_mode=False,
    train_percentage_unknowns=0.5,
):
    data = HeteroData()

    global_to_local, local_to_global, local_node_counts = global_local_matcher(
        nodes_ids
    )
    save_index_mapping(global_to_local, local_to_global)

    for node_type in nodes_ids.keys():
        anchors, is_anchor, is_unknown = identify_anchors(
            nodes_features[node_type], train_mode, train_percentage_unknowns
        )

        # manage node ids
        local_ids = list(range(local_node_counts[node_type]))
        ids = torch.tensor(local_ids).unsqueeze(1).float()  # [num_nodes, 1]

        # vectorize features
        vectorized_features = vectorize_features(nodes_features[node_type])
        features = torch.tensor(vectorized_features).float()  # [num_nodes, feature_dim]
        x = torch.cat([ids, features], dim=1)
        data[node_type].x = x

        # add anchors as a feature
        is_anchor_feat = is_anchor.view(-1, 1).float()
        anchor_feat = anchors.view(-1, 1).float()
        anchor_feats = torch.cat([is_anchor_feat, anchor_feat], dim=1)  # [num_nodes, 2]
        data[node_type].x = torch.cat(
            [data[node_type].x, anchor_feats], dim=1
        )  # [num_nodes, feature_dim]

        data[node_type].y = anchors
        data[node_type].is_anchor = is_anchor
        data[node_type].is_unknown = is_unknown

    for rel_type, subdict in edges_indices.items():
        for meta_type, edge_arr in subdict.items():
            source, target = (
                relationships[rel_type][meta_type]["source"],
                relationships[rel_type][meta_type]["target"],
            )
            src_ids = [global_to_local[source][sid] for sid in edge_arr[0]]
            dst_ids = [global_to_local[target][tid] for tid in edge_arr[1]]

            edge_index_tensor = torch.tensor([src_ids, dst_ids], dtype=torch.long)
            data[(source, rel_type.lower(), target)].edge_index = edge_index_tensor
    data = data_transformer(data)
    return data
