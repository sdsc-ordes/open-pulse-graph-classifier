from graphdatascience import ServerVersion


def cora_loading(gds):
    assert gds.server_version() >= ServerVersion(2, 5, 0)
    G = gds.graph.load_cora()
    print(f"Metadata for our loaded Cora graph `G`: {G}")
    print(f"Node labels present in `G`: {G.node_labels()}")
    return G
