import torch
from neo4jdownloader import Neo4JDownloader
from data_processor import create_heterogenous_data, add_labels
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file


def load_saved_data():
    try:
        data = torch.load("open-pulse-graph-classifier/data/heteoro_data.pt")
        return data
    except FileNotFoundError:
        print("No saved data found. Extracting data from Neo4j.")
        return None


def extract_data():
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
        torch.save(data, "open-pulse-graph-classifier/data/heteoro_data.pt")
        return data
    finally:
        downloader.close()
