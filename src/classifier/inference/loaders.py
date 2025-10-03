from classifier.processing.loaders import build_loader


def make_inference_loader(data, batch_size=32, num_neighbors=[10, 5], random_state=42):
    loaders = {}

    for ntype in data.node_types:
        node_store = data[ntype]

        # Split ONLY unknown nodes into train/val/test (if they exist)
        if hasattr(node_store, "is_unknown") and node_store.is_unknown.any():
            unknown_idx = node_store.is_unknown.nonzero(as_tuple=True)[0].cpu().numpy()
            inference_loader = build_loader(data, ntype, unknown_idx, shuffle=False)
            loaders[ntype] = inference_loader

    return loaders
