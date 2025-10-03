import torch
from torch_geometric.loader import NeighborLoader
from huggingface_hub import hf_hub_download
import json

from classifier.processing.data_extraction import (
    get_downloader,
    extract_data,
)
from classifier.processing.data_transformer import data_transformer
from classifier.inference.loaders import make_inference_loader
from classifier.processing.predictions_upload import upload_to_neo4j

from classifier.huggingface.hf_download_model import download_model_from_huggingface


@torch.no_grad()
def inference(neo4j_database):
    extracted_data = extract_data(neo4j_database)
    print(
        "Validating data with PyG tools (data.validate()):", extracted_data.validate()
    )

    # download model
    download_model_from_huggingface()

    # load model
    loaded_model = torch.load("open-pulse-graph-classifier/models/supervised_hetero.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loaded_model.to(device)
    model.eval()

    # inference
    loaders = make_inference_loader(extracted_data)
    inference_loaders = {ntype: loader[0] for ntype, loader in loaders.items()}

    # all probs needs to save id and prediction
    all_probs = {ntype: [] for ntype in inference_loaders.keys()}
    with torch.no_grad():
        for ntype, loader in inference_loaders.items():
            for batch in loader:
                batch = batch.to(device)
                logits_dict = model(batch.x_dict, batch.edge_index_dict)
                if hasattr(batch[ntype], "batch_size"):
                    batch_size = batch[ntype].batch_size
                    if batch_size == 0:
                        continue
                    logits = logits_dict[ntype][:batch_size].view(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    nodes_probs = [
                        {nodeid: prob}
                        for nodeid, prob in zip(
                            batch[ntype].node_id[:batch_size].cpu().numpy(), probs
                        )
                    ]
                    all_probs[ntype].extend(nodes_probs)

    # upload to neo4j
    upload_to_neo4j(all_probs, neo4j_database)
    return
