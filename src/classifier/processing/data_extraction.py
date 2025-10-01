import torch
from classifier.neo4j.neo4jdownloader import Neo4JDownloader
from classifier.processing.data_heterogenous import (
    create_heterogenous_data,
)
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file


def load_saved_data():
    try:
        data = torch.load(
            "open-pulse-graph-classifier/data/heteoro_data.pt", weights_only=False
        )
        return data
    except FileNotFoundError:
        print("No saved data found. Extracting data from Neo4j.")
        return None


def get_downloader(neo4j_database):
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USERNAME = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    NEO4J_DATABASE = neo4j_database

    return Neo4JDownloader(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)


def extract_data(neo4j_database=None, train_mode=False, train_percentage_unknowns=0.5):
    downloader = get_downloader(neo4j_database)

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
            "type": {"source": "org", "target": "repo"},
        },
    }

    try:
        # downloader.retrieve_all()
        nodes_ids, nodes_features = downloader.retrieve_nodes(nodes)
        edges_indices, edges_attributes = downloader.retrieve_edges(relationships)
        # print(nodes_ids["org"])
        # print(nodes_features["org"])
        # print(edges_indices)

        data = create_heterogenous_data(
            nodes_ids,
            nodes_features,
            edges_indices,
            relationships,
            train_mode,
            train_percentage_unknowns,
        )
        # torch.save(data, "open-pulse-graph-classifier/data/heteoro_data.pt")
        return data
    finally:
        downloader.close()
