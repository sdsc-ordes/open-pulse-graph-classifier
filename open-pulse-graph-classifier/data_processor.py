import torch
import tensorflow as tf
from torch_geometric.data import HeteroData


def create_tensor_matrix(array1, array2):
    # +1 to match the index augmentation of 1 of the nodes
    # to avoid having a zero index node
    tensor1 = torch.tensor(array1 + 1)
    tensor2 = torch.tensor(array2 + 1)
    tensor_matrix = torch.stack((tensor1, tensor2))
    return tensor_matrix


def create_heterogenous_data(nodes_ids, edges_indices, relationships):
    data = HeteroData()

    for key, val in nodes_ids.items():
        # +1 augmentation of the indices of the nodes to avoid having a zero index node
        data[key].x = tf.convert_to_tensor([v + 1 for v in val])

    for relationship, subdict in relationships.items():
        for type, definition in subdict.items():
            source = definition["source"]
            target = definition["target"]
            # if that relationship exists in the graph
            if len(edges_indices[relationship][type]) > 0:
                data[source, relationship, target].edge_index = create_tensor_matrix(
                    edges_indices[relationship][type][0],
                    edges_indices[relationship][type][1],
                )
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
