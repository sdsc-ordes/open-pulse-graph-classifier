import torch
from torch_geometric.data import HeteroData
import json


def global_local_matcher(nodes_ids):
    global_to_local = {}
    local_node_counts = {}

    for node_type, ids in nodes_ids.items():
        global_to_local[node_type] = {gid: i for i, gid in enumerate(ids)}
        local_node_counts[node_type] = len(ids)

    local_to_global = {
        node_type: {i: gid for i, gid in enumerate(gid_list)}
        for node_type, gid_list in nodes_ids.items()
    }
    return global_to_local, local_to_global, local_node_counts


def save_index_mapping(global_to_local, local_to_global):
    with open("open-pulse-graph-classifier/data/global_to_local.json", "w") as fp:
        json.dump(global_to_local, fp)
    with open("open-pulse-graph-classifier/data/local_to_global.json", "w") as fp:
        json.dump(local_to_global, fp)


def vectorize_features(features):
    from sklearn.feature_extraction import DictVectorizer

    # TO-DO: come back and see if this is the right way to vectorize features
    # is this correct or do we need to save it to use the same all the time?
    # is this the right technique?
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(features).toarray()
    return features_vectorized


def create_heterogenous_data(nodes_ids, nodes_features, edges_indices, relationships):
    data = HeteroData()

    global_to_local, local_to_global, local_node_counts = global_local_matcher(
        nodes_ids
    )
    save_index_mapping(global_to_local, local_to_global)

    for node_type in nodes_ids.keys():
        vectorized_features = vectorize_features(nodes_features[node_type])
        local_ids = list(range(local_node_counts[node_type]))
        ids = torch.tensor(local_ids).unsqueeze(1).float()  # [num_nodes, 1]
        features = torch.tensor(vectorized_features).float()  # [num_nodes, feature_dim]
        x = torch.cat([ids, features], dim=1)
        data[node_type].x = x

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
    return data


def add_labels(data, label):
    node_types, _ = data.metadata()
    for node_type in node_types:
        num_nodes = data[node_type].x.shape[0]
        data[node_type].y = torch.tensor([label] * num_nodes)
    return data
