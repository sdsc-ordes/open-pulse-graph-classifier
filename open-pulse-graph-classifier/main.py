# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# from torch_geometric.transforms import RandomNodeSplitter

import os
from neo4jdownloader import Neo4JDownloader

if __name__ == "__main__":
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
    NEO4J_USERNAME = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

    nodes = ["User", "Repository", "Organisation"]
    relationships = {
        "MEMBER_OF": {"source": "User", "target": "Organisation"},
        "OWNER_OF": {"source": "User", "target": "Repository"},
        "CONTRIBUTOR_OF": {"source": "User", "target": "Repository"},
        "FORK_OF": {"source": "Repository", "target": "Repository"},
    }

    downloader = Neo4JDownloader(
        NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
    )

    try:
        nodes_ids, nodes_features = downloader.retrieve_nodes(nodes)
        edges_index, edges_attributes = downloader.retrieve_edges(relationships)
        print(nodes_ids)
        print(nodes_features)
    finally:
        downloader.close()
