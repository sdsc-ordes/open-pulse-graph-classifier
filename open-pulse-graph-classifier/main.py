import os
import torch
from torch_geometric.nn import to_hetero


from data_extraction import load_saved_data, extract_data
from data_transformer import data_transformer
from models.supervised import GNN
from loaders import split_data
from train_eval import train, evaluate

if __name__ == "__main__":
    data = load_saved_data()
    if not data:
        data = extract_data()

    if data:
        # transform data
        data = data_transformer(data)
        # print("Full data:")
        # print(data)
        # print(data['user', 'member of', 'org'].edge_index)
        # print(data.validate())

        # split data
        train_loader, test_loader, val_loader = split_data(data)

        # create model
        model_supervised = GNN(hidden_channels=64, out_channels=2)
        model_supervised_hetero = to_hetero(
            model_supervised, data.metadata(), aggr="sum"
        )

        # # train model
        n_epochs = 100
        # try to change cuda to mps
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_supervised_hetero.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        loss = train(train_loader, device, model, optimizer, n_epochs)

        # # evaluate model
        # results = evaluate(test_loader, device, model)
        # for node_type in data.node_types:
        #     print(
        #         f"Node Type {node_type} has accuracy of {results[node_type]['accuracy']} and AUC score of {results[node_type]['roc_auc']}"
        #     )
