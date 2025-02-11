from neo4j import GraphDatabase, RoutingControl
from neo4j.exceptions import DriverError, Neo4jError
import logging
import numpy as np


class Neo4JDownloader:
    def __init__(self, uri, user, password, database=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        # Don't forget to close the driver connection when you are finished
        # with it
        self.driver.close()

    def get_nodes(self, driver, label):
        query = f"""
        MATCH (n:{label})
        RETURN ID(n) AS id, n.feature_vector AS features
        """
        try:
            results = driver.run(query)
            node_ids, features = [], []
            for record in results:
                node_ids.append(record["id"])
                features.append(
                    record["features"]
                )  # Assuming features are stored as a list
            print("GET NODES")
            print(node_ids)
            return node_ids, np.array(features, dtype=np.float32)
        except (DriverError, Neo4jError) as exception:
            logging.error("%s raised an error: \n%s", query, exception)
            raise

    def get_edges(self, driver, src_label, rel_type, dst_label):
        query = f"""
        MATCH (a:{src_label})-[r:{rel_type}]->(b:{dst_label})
        RETURN ID(a) AS src, ID(b) AS dst, r.edge_features AS edge_features
        """
        results = driver.run(query)
        edge_index, edge_attrs = [], []
        for record in results:
            edge_index.append([record["src"], record["dst"]])
            edge_attrs.append(record["edge_features"])
        return np.array(edge_index).T, np.array(edge_attrs, dtype=np.float32)

    def retrieve_nodes(self, nodes_list):
        ids = {}
        feats = {}
        with self.driver.session() as session:
            for node in nodes_list:
                id, feat = session.execute_read(self.get_nodes, node)
                ids[node] = id
                feats[node] = feat
        return ids, feats

    def retrieve_edges(self, relationship_dict):
        edges_index = {}
        edges_attributes = {}
        for key, val in relationship_dict.items():
            with self.driver.session() as session:
                source = val["source"]
                target = val["target"]
                relationship = key
                edge_index, edge_attributes = session.execute_read(
                    self.get_edges, source, relationship, target
                )
                edges_index[key] = edge_index
                edges_attributes[key] = edge_attributes
        return edges_index, edges_attributes
