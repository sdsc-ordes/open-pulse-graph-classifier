from neo4j import GraphDatabase, RoutingControl
from neo4j.exceptions import DriverError, Neo4jError
import logging


class App:
    def __init__(self, uri, user, password, database=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        # Don't forget to close the driver connection when you are finished
        # with it
        self.driver.close()

    def upload_graph(self, file_path):
        query = f"""

            LOAD CSV WITH HEADERS FROM 'file:///{file_path}' AS row

            // Create Source Node
            MERGE (s {{name: row.source}})
            FOREACH (_ IN CASE WHEN row.source_type = 'user' THEN [1] ELSE [] END | SET s:User)
            FOREACH (_ IN CASE WHEN row.source_type = 'org' THEN [1] ELSE [] END | SET s:Organisation)
            FOREACH (_ IN CASE WHEN row.source_type = 'repo' THEN [1] ELSE [] END | SET s:Repository)

            // Create Target Node
            MERGE (t {{name: row.target}})
            FOREACH (_ IN CASE WHEN row.target_type = 'user' THEN [1] ELSE [] END | SET t:User)
            FOREACH (_ IN CASE WHEN row.target_type = 'org' THEN [1] ELSE [] END | SET t:Organisation)
            FOREACH (_ IN CASE WHEN row.target_type = 'repo' THEN [1] ELSE [] END | SET t:Repository)

            // Create Bidirectional Relationship
            MERGE (s)-[r1:CONNECTED_TO]->(t)
            SET r1.property = row.property

            MERGE (t)-[r2:CONNECTED_TO]->(s)
            SET r2.property = row.property;


            """
        try:
            record, summary, key = self.driver.execute_query(
                query, params={}, database_=self.database
            )
            return summary
        # Capture any errors along with the query and data for traceability
        except (DriverError, Neo4jError) as exception:
            logging.error("%s raised an error: \n%s", query, exception)
            raise
