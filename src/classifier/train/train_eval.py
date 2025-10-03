import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from itertools import cycle
import numpy as np


def train(loaders_dict, device, model, optimizer, n_epochs):
    """
    loaders_dict: dict {ntype: train_loader}
    """
    print(f"Training jointly on node types: {list(loaders_dict.keys())}")
    loss_per_epoch = []
    # this could be a list if different node types need different criterion
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        steps = 0

        # iterate in round-robin fashion over all loaders
        iterators = {ntype: cycle(loader) for ntype, loader in loaders_dict.items()}
        # Each epoch goes for as long as the biggest loader.
        # Smaller loaders will cycle through their data multiple times.
        num_batches = max(len(loader) for loader in loaders_dict.values())

        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_losses = []

            print("Processing batch...")

            for ntype, iterator in iterators.items():
                print(f"Processing node type {ntype}")

                batch = next(iterator).to(device)
                logits_dict = model(batch.x_dict, batch.edge_index_dict)

                if hasattr(batch[ntype], "y") and hasattr(batch[ntype], "batch_size"):
                    logits = logits_dict[ntype][: batch[ntype].batch_size].view(-1)
                    y = batch[ntype].y[: batch[ntype].batch_size].float()

                    loss = criterion(logits, y)
                    batch_losses.append(loss)

            if batch_losses:
                loss = sum(batch_losses)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

        epoch_loss = total_loss / max(steps, 1)
        loss_per_epoch.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")

    return loss_per_epoch


@torch.no_grad()
def evaluate(loaders_dict, device, model):
    """
    loaders_dict: dict {ntype: val_loader}
    """
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    steps = 0

    all_probs = {ntype: [] for ntype in loaders_dict.keys()}
    all_labels = {ntype: [] for ntype in loaders_dict.keys()}

    with torch.no_grad():
        for ntype, val_loader in loaders_dict.items():
            for batch in val_loader:
                batch = batch.to(device)
                logits_dict = model(batch.x_dict, batch.edge_index_dict)

                if hasattr(batch[ntype], "y") and hasattr(batch[ntype], "batch_size"):
                    batch_size = batch[ntype].batch_size
                    if batch_size == 0:
                        continue

                    logits = logits_dict[ntype][:batch_size].view(-1)
                    y = batch[ntype].y[:batch_size].float()

                    loss = criterion(logits, y)
                    total_loss += loss.item()
                    steps += 1

                    probs = torch.sigmoid(logits).cpu().numpy()
                    labels = y.cpu().numpy()
                    all_probs[ntype].append(probs)
                    all_labels[ntype].append(labels)

    # Aggregate metrics
    results = {}
    for ntype in loaders_dict.keys():
        if len(all_probs[ntype]) == 0:
            continue
        probs = np.concatenate(all_probs[ntype])
        labels = np.concatenate(all_labels[ntype])
        acc = accuracy_score(labels, (probs >= 0.5).astype(int))
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")  # e.g., only one class present
        results[ntype] = {"accuracy": acc, "roc_auc": auc, "n": len(labels)}

    avg_loss = total_loss / max(steps, 1)

    return avg_loss, results
