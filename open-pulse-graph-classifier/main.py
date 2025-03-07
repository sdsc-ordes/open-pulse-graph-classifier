import os
import torch
from torch_geometric.nn import to_hetero

from neo4jdownloader import Neo4JDownloader
from data_processor import create_heterogenous_data, add_labels
from data_transformer import data_transformer
from models.supervised import GNN
from loaders import split_data
from train_eval import train, evaluate

if __name__ == "__main__":
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
    NEO4J_USERNAME = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

    nodes = ["user", "repo", "org"]
    relationships = {
        "member of": {"type1": {"source": "user", "target": "org"}},
        "owner of": {
            "type1": {"source": "user", "target": "repo"},
            "type2": {"source": "org", "target": "repo"},
        },
        "contributor of": {
            "type1": {"source": "user", "target": "repo"},
            "type2": {"source": "org", "target": "repo"},
        },
        "fork of": {
            "type1": {"source": "user", "target": "repo"},
            "type1": {"source": "org", "target": "repo"},
        },
    }

    downloader = Neo4JDownloader(
        NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
    )

    try:
        # downloader.retrieve_all()
        nodes_ids, nodes_features = downloader.retrieve_nodes(nodes)
        edges_indices, edges_attributes = downloader.retrieve_edges(relationships)
        # print(nodes_ids["org"])
        # print(nodes_features["org"])
        # print(edges_indices)

        data = create_heterogenous_data(nodes_ids, edges_indices, relationships)
        data = add_labels(data, 1)

    finally:
        downloader.close()

    if data:
        # transform data
        data = data_transformer(data)
        # split data
        train_loaders, test_loaders, val_loaders = split_data(data)

        # create model
        model_supervised = GNN(hidden_channels=64, out_channels=2)
        model_supervised_hetero = to_hetero(
            model_supervised, data.metadata(), aggr="sum"
        )

        # train model
        n_epochs = 100
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_supervised_hetero.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        loss = train(train_loaders, device, model, optimizer, n_epochs)

        # evaluate model
        results = evaluate(test_loaders, device, model)
        for node_type in data.node_types:
            print(
                f"Node Type {node_type} has accuracy of {results[node_type]['accuracy']} and AUC score of {results[node_type]['roc_auc']}"
            )
