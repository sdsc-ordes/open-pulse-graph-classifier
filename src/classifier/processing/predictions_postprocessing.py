import torch.nn.functional as F
import json


def get_mapping_local_to_global():
    with open("src/classifier/data_mapper/local_to_global.json", "r") as fp:
        local_to_global = json.load(fp)
    return local_to_global


def get_nodeID(node_id, node_type, local_to_global):
    global_id = local_to_global[node_type][str(node_id)]
    return global_id


def predictions_map_format(all_probs):
    # builds predictions for neo4j
    # output format:
    # predictions = [
    #     {"id": 1, "prediction": 0.42},
    #     {"id": 2, "prediction": 0.99},
    #     {"id": 3, "prediction": 0.12}
    # ]
    predictions = []
    local_to_global = get_mapping_local_to_global()
    for ntype, probs_list in all_probs.items():
        for prob_dict in probs_list:
            for i, prob in prob_dict.items():
                node_id = i
                global_id = get_nodeID(node_id, ntype, local_to_global)
                predictions.append({"id": global_id, "prediction": float(prob)})
    return predictions
