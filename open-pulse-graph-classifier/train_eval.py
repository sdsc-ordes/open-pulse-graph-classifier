import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def train(train_loaders, device, model, optimizer, n_epochs):
    """
    Train function that supports multiple train_loaders (one per node type) for multiple epochs.

    Args:
        model: The GNN model.
        optimizer: Optimizer (e.g., Adam).
        device: Device (CPU/GPU).
        train_loaders: Dictionary of DataLoaders, one per node type.
        n_epochs (int): Number of training epochs.

    Returns:
        list: List of average loss values for each epoch.
    """
    loss_per_epoch = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_batches = 0

        for node_type, loader in train_loaders.items():
            for batch in loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                print(batch.edge_index_dict)
                out = model(batch.x_dict, batch.edge_index_dict)

                # Compute loss for the current node type
                if node_type in batch:
                    loss = F.cross_entropy(out, batch[node_type].y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        loss_per_epoch.append(avg_loss)

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return loss_per_epoch


@torch.no_grad()
def evaluate(loaders, device, model):
    """
    Evaluates model accuracy and AUC-ROC per node type.

    Args:
        loaders (dict): A dictionary of DataLoaders, one per node type.
        model: The trained GNN model.
        device: The device (CPU/GPU).

    Returns:
        dict: Dictionary with accuracy and AUC-ROC per node type.
    """
    model.eval()
    results = {}

    for node_type, loader in loaders.items():
        correct, total = 0, 0
        all_preds, all_labels = [], []

        for batch in loader:
            batch = batch.to(device)

            if node_type in batch:
                out = model(batch.x_dict, batch.edge_index_dict)
                pred = out.argmax(dim=1)
                y_true = batch[node_type].y

                correct += (pred == y_true).sum().item()
                total += y_true.size(0)

                all_preds.append(
                    out.softmax(dim=1).cpu()
                )  # Convert logits to probabilities
                all_labels.append(y_true.cpu())

        # Compute accuracy
        accuracy = correct / total if total > 0 else 0

        if total > 1:
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()
            auc = roc_auc_score(
                all_labels, all_preds, multi_class="ovr"
            )  # One-vs-Rest AUC for multi-class
        else:
            auc = None

        results[node_type] = {"accuracy": accuracy, "roc_auc": auc}

    return results
