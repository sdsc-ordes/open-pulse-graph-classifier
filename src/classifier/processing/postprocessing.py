import torch.nn.functional as F
import json


def get_probs_preds(logits):
    # logits are [N_nodes, n_classes]
    preds = logits.argmax(dim=1)
    probs = F.softmax(logits, dim=1)[:, 1]  # Prob for class 1
    return preds, probs


def get_node_name(node_id, node_type, downloader):
    with open("open-pulse-graph-classifier/data/local_to_global.json", "r") as fp:
        local_to_global = json.load(fp)
    global_id = local_to_global[node_type][node_id]
    name = downloader.get_node_name_by_id(global_id)
    return name


def get_positive_output(nodes, node_type, probs, threshold, downloader):
    positive_output = {}
    for prob in probs:
        if prob >= threshold:
            node_id = nodes[probs.index(prob)]
            node_name = get_node_name(node_id, node_type, downloader)
            positive_output.update({node_name: prob})
    return positive_output
