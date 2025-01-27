import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomNodeSplit

from open_pulse_graph_classifier.cora_loader import cora_loading
from open_pulse_graph_classifier.neo4j_connection import connect_neo4j

# Set seeds for consistent results
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

gds = connect_neo4j()
G = cora_loading(gds)

sample_topology_df = gds.graph.relationships.stream(G)
display(sample_topology_df)
sample_topology = sample_topology_df.by_rel_type()
print(f"Relationship type keys: {sample_topology.keys()}")
