import torch
from torch_geometric.data import HeteroData
import json


def create_tensor_matrix(array1, array2):
    # +1 to match the index augmentation of 1 of the nodes
    # to avoid having a zero index node
    tensor1 = torch.tensor(array1 + 1)
    tensor2 = torch.tensor(array2 + 1)
    tensor_matrix = torch.stack((tensor1, tensor2))
    return tensor_matrix


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


def create_heterogenous_data(nodes_ids, edges_indices, relationships):
    data = HeteroData()

    global_to_local, local_to_global, local_node_counts = global_local_matcher(
        nodes_ids
    )
    save_index_mapping(global_to_local, local_to_global)
    for node_type in nodes_ids.keys():
        local_ids = list(range(local_node_counts[node_type]))
        x = torch.tensor(local_ids).unsqueeze(1).float()  # or use your real features
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

    # for key, val in nodes_ids.items():
    #     # +1 augmentation of the indices of the nodes to avoid having a zero index node
    #     data[key].x = torch.Tensor([v + 1 for v in val])

    # for relationship, subdict in relationships.items():
    #     for type, definition in subdict.items():
    #         source = definition["source"]
    #         target = definition["target"]
    #         # if that relationship exists in the graph
    #         if len(edges_indices[relationship][type]) > 0:
    #             data[source, relationship, target].edge_index = create_tensor_matrix(
    #                 edges_indices[relationship][type][0],
    #                 edges_indices[relationship][type][1],
    #             )
    return data


def add_labels(data, label):
    node_types, _ = data.metadata()
    for node_type in node_types:
        num_nodes = data[node_type].x.shape[0]
        data[node_type].y = torch.tensor([label] * num_nodes)
    return data


# from sklearn import preprocessing
# features are strings, they need to be encoded
# label_encoder = preprocessing.LabelEncoder()
# nodes_features_encoded = {}
# for node, features in nodes_features.items():
#     nodes_features_encoded[node] = encode(label_encoder, features)
# def encode(label_encoder, feat_string):
#      # feat_String is a list of strings
#     feat_encoded = label_encoder.fit_transform(feat_string)
#     return feat_encoded
