from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
import logging
import numpy as np


class Neo4JUploader:
    def __init__(self, uri, user, password, database=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        # Don't forget to close the driver connection when you are finished
        # with it
        self.driver.close()

    def upload_predictions(self, driver, predictions):
        # example data
        # predictions = [
        #     {"id": 1, "prediction": 42},
        #     {"id": 2, "prediction": 99},
        #     {"id": 3, "prediction": 123}
        # ]
        query = """
            UNWIND $predictions AS predictionData
            MATCH (n:MyNode { id: predictionData.id })
            SET n.prediction = predictionData.value
            RETURN n
            """
        driver.run(query, {"predictions": predictions})

    def upload_nodes_predictions(self, predictions):
        try:
            with self.driver.session(database=self.database) as session:
                session.write_transaction(self.upload_predictions, predictions)
        except (DriverError, Neo4jError) as exception:
            logging.error(
                "An error occurred while uploading predictions: \n%s", exception
            )
            raise
