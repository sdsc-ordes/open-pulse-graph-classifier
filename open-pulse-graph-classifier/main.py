import os
from neo4jdownloader import Neo4JDownloader
from data_processor import create_heterogenous_data

if __name__ == "__main__":
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
    NEO4J_USERNAME = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

    nodes = ["User", "Repository", "Organisation"]
    relationships = {
        "MEMBER_OF": {"type1": {"source": "User", "target": "Organisation"}},
        "OWNER_OF": {
            "type1": {"source": "User", "target": "Repository"},
            "type2": {"source": "Organisation", "target": "Repository"},
        },
        "CONTRIBUTOR_OF": {
            "type1": {"source": "User", "target": "Repository"},
            "type2": {"source": "Organisation", "target": "Repository"},
        },
        "FORK_OF": {
            "type1": {"source": "User", "target": "Repository"},
            "type1": {"source": "Organisation", "target": "Repository"},
        },
    }

    downloader = Neo4JDownloader(
        NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
    )

    try:
        # downloader.retrieve_all()
        nodes_ids, nodes_features = downloader.retrieve_nodes(nodes)
        edges_indices, edges_attributes = downloader.retrieve_edges(relationships)
        # print(nodes_ids)
        # print(nodes_features)
        # print(edges_indices)

        data = create_heterogenous_data(nodes_ids, edges_indices, relationships)
        print(data)
    finally:
        downloader.close()
