import torch
from torch_geometric.nn import to_hetero
from huggingface_hub import HfApi

from classifier.processing.data_extraction import extract_data
from classifier.models.supervised import GNN
from classifier.train.loaders import make_loaders
from classifier.train.train_eval import train, evaluate


def training(neo4j_database, train_percentage_unknowns=0.5):
    train_mode = True
    data = extract_data(neo4j_database, train_mode, train_percentage_unknowns)

    if data:
        # transform data

        # print("Full data:")
        # print(data)
        # print(data['user', 'member of', 'org'].edge_index)
        print("Validating data with PyG tools (data.validate()):", data.validate())

        loaders = make_loaders(data)

        # extract train loaders for all node types
        train_loaders = {ntype: loader[0] for ntype, loader in loaders.items()}

        model_supervised = GNN(hidden_channels=64, out_channels=2)
        model_supervised_hetero = to_hetero(
            model_supervised, data.metadata(), aggr="sum"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_supervised_hetero.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        loss = train(train_loaders, device, model, optimizer, n_epochs=1)

        # save model
        torch.save(model, "classifier/models/supervised_hetero.pt")
        print("Model saved to classifier/models/supervised_hetero.pt")
        api = HfApi()
        api.upload_file(
            path_or_fileobj="classifier/models/supervised_hetero.pt",
            path_in_repo="classifier/models/supervised_hetero.pt",
            repo_id="SDSC/open-pulse-graph-classifier",
            repo_type="model",
        )

        # evaluate model
        test_loaders = {ntype: loader[1] for ntype, loader in loaders.items()}
        val_loaders = {ntype: loader[2] for ntype, loader in loaders.items()}
        results = evaluate(test_loaders, device, model)
        print(results)
        for node_type in data.node_types:
            print(
                f"Node Type {node_type} has accuracy of {results[node_type]['accuracy']} and AUC score of {results[node_type]['roc_auc']}"
            )

        # ----------------------------------
        # TEST NEO4J UPLOAD
        # from classifier.processing.predictions_upload import upload_to_neo4j

        # all_probs = fake_all_probs(train_loaders)
        # upload_to_neo4j(all_probs, neo4j_database)


# ----------------------------------
# TEST NEO4J UPLOAD
# def fake_all_probs(loaders):
#     import json

#     # using the local_to_global mapping create a fake all_probs dict for part of the data
#     all_probs = {ntype: [] for ntype in loaders.keys()}
#     with open("classifier/data_mapper/local_to_global.json", "r") as fp:
#         local_to_global = json.load(fp)
#     for ntype, mapping in local_to_global.items():
#         num_nodes = len(mapping)
#         # create fake probabilities
#         probs = torch.rand(num_nodes).numpy()
#         nodes_probs = [
#             {nodeid: float(prob)} for nodeid, prob in zip(range(num_nodes), probs)
#         ]
#         all_probs[ntype].extend(nodes_probs)
#     return all_probs


if __name__ == "__main__":
    # TO-DO remove hard coded. add an env variable?
    neo4j_database = "neo4j"
    train_percentage_unknowns = 0.5
    training(neo4j_database, train_percentage_unknowns)
