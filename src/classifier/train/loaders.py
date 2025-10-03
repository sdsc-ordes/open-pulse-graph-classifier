from sklearn.model_selection import train_test_split

from classifier.processing.loaders import build_loader


def make_loaders(data, batch_size=32, num_neighbors=[10, 5], random_state=42):
    loaders = {}

    for ntype in data.node_types:
        node_store = data[ntype]

        # Split ONLY unknown nodes into train/val/test (if they exist)
        if hasattr(node_store, "is_unknown") and node_store.is_unknown.any():
            unknown_idx = node_store.is_unknown.nonzero(as_tuple=True)[0].cpu().numpy()
            y_unknown = node_store.y[node_store.is_unknown].cpu().numpy()

            train_idx, testval_idx = train_test_split(
                unknown_idx,
                test_size=0.4,
                random_state=random_state,
                stratify=y_unknown,
            )
            y_testval = node_store.y[testval_idx].cpu().numpy()
            val_idx, test_idx = train_test_split(
                testval_idx,
                test_size=0.5,
                random_state=random_state,
                stratify=y_testval,
            )

            train_loader = build_loader(data, ntype, train_idx, shuffle=True)
            test_loader = build_loader(data, ntype, test_idx, shuffle=False)
            val_loader = build_loader(data, ntype, val_idx, shuffle=False)

            loaders[ntype] = (train_loader, test_loader, val_loader)

    return loaders
