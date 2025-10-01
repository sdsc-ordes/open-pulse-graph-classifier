import torch_geometric.transforms as T
import numpy as np
import torch
import json
from sklearn.feature_extraction import DictVectorizer


def data_transformer(data):
    # Transofrm data: make undirected and normalize
    data = T.ToUndirected()(data)
    # data = T.NormalizeFeatures()(data)
    return data


def identify_anchors(node_features, train_mode=False, train_percentage_unknowns=0.5):
    # we need to start from nodes_features and based on anchor / labels we make a mask in data.
    # we need a train parameter so that in training we arbitrarily remove set as unknown a bunch of nodes while the other remain anchors
    # raw property: 1=part of community, 0=not part of community, -1=unknown
    anchors = [feat["anchor"] for feat in node_features]
    anchors = np.array(anchors)

    # for training we artificially set to -1 a certain percentage
    if train_mode:
        anchors = train_remove_anchors(anchors, train_percentage_unknowns)

    # Anchor: must have valid label (0 or 1)
    is_anchor = anchors != -1
    # Unknown = no label (y == -1)
    is_unknown = anchors == -1

    anchors = torch.from_numpy(anchors)
    is_anchor = torch.from_numpy(is_anchor)
    is_unknown = torch.from_numpy(is_unknown)

    return anchors, is_anchor, is_unknown


def train_remove_anchors(anchors, percentage):
    # in training, we know all labels. But we need to keep some for anchors and remove the other ones.
    num_to_replace = int(len(anchors) * percentage)
    indices = np.random.choice(len(anchors), num_to_replace, replace=False)
    anchors[indices] = -1
    return anchors


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
    with open("src/classifier/data_mapper/global_to_local.json", "w") as fp:
        json.dump(global_to_local, fp)
    with open("src/classifier/data_mapper/local_to_global.json", "w") as fp:
        json.dump(local_to_global, fp)


def vectorize_features(features):
    # TO-DO: come back and see if this is the right way to vectorize features
    # is this correct or do we need to save it to use the same all the time?
    # is this the right technique?
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(features).toarray()
    return features_vectorized
