import torch
from dotenv import load_dotenv
import os

from classifier.neo4j.neo4juploader import Neo4JUploader
from classifier.processing.predictions_postprocessing import predictions_map_format


def get_uploader(neo4j_database):
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USERNAME = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    NEO4J_DATABASE = neo4j_database

    return Neo4JUploader(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)


def upload_to_neo4j(all_probs, neo4j_database=None):
    neo4j_predictions = predictions_map_format(all_probs)
    uploader = get_uploader(neo4j_database)

    try:
        uploader.upload_nodes_predictions(neo4j_predictions)
        print("Predictions successfully uploaded to Neo4j.")
    except Exception as e:
        print(f"An error occurred while uploading predictions: {e}")
    finally:
        uploader.close()
