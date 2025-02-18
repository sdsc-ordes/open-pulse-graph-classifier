import torch_geometric.transforms as T


def data_transformer(data):
    # Transofrm data: make undirected and normalize
    data = T.ToUndirected()(data)
    data = T.NormalizeFeatures()(data)
    return data
