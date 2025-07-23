import torch
from torch_geometric.loader import NeighborLoader
from huggingface_hub import hf_hub_download
import json


from classifier.processing.data_extraction import (
    get_downloader,
    extract_data,
)
from classifier.processing.data_transformer import data_transformer
from classifier.processing.postprocessing import (
    get_probs_preds,
    get_positive_output,
)


@torch.no_grad()
def inference(neo4j_database):
    extracted_data = extract_data(neo4j_database)
    transformed_data = data_transformer(extracted_data)

    # download model
    # TO-DO: model name should become a parameter in the future
    hf_hub_download(
        repo_id="SDSC/open-pulse-graph-classifier",
        filename="models/supervised_hetero.pt",
        local_dir="open-pulse-graph-classifier/models/supervised_hetero.pt",
    )

    # load model
    loaded_model = torch.load("open-pulse-graph-classifier/models/supervised_hetero.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loaded_model.to(device)
    model.eval()

    # inference
    # TO-DO: threshold should become a parameter in the future
    threshold = 0.75
    inference_loader = NeighborLoader(
        transformed_data,
        input_nodes=("user", transformed_data["user"]),
        num_neighbors=[15, 10],
        batch_size=100,
        shuffle=False,
    )
    downloader = get_downloader(neo4j_database)
    output = {}
    for batch in inference_loader:
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch.x_dict, batch.edge_index_dict)
            for node_type in batch.node_types:
                if node_type not in out or node_type not in batch.node_types:
                    continue
                print("Processing output for node type:", node_type)
                logits = out[node_type]  # [N_nodes, n_classes]
                _, probs = get_probs_preds(logits)
                output[node_type].update(
                    get_positive_output(
                        batch.x_dict[node_type], node_type, probs, threshold, downloader
                    )
                )
    return output
