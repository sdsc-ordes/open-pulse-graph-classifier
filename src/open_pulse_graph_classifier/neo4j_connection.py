from graphdatascience import GraphDataScience
import os


def connect_neo4j():
    # Get Neo4j DB URI, credentials and name from environment if applicable
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_AUTH = None
    NEO4J_DB = os.environ.get("NEO4J_DB", "neo4j")
    if os.environ.get("NEO4J_USER") and os.environ.get("NEO4J_PASSWORD"):
        NEO4J_AUTH = (
            os.environ.get("NEO4J_USER"),
            os.environ.get("NEO4J_PASSWORD"),
        )
    gds = GraphDataScience(NEO4J_URI, auth=NEO4J_AUTH, database=NEO4J_DB)
    return gds
