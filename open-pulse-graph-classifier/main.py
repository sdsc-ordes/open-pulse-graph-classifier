# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# from torch_geometric.transforms import RandomNodeSplitter

import os
from app import App

if __name__ == "__main__":
    file_path = "data.csv"
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
    NEO4J_USERNAME = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

    print(NEO4J_USERNAME)
    print(NEO4J_PASSWORD)
    app = App(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
    try:
        summary = app.upload_graph(file_path)
        print(summary)
    finally:
        app.close()
