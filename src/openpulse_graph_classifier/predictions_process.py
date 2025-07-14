import torch.nn.functional as F


def get_probs_preds(logits):
    # logits are [N_nodes, n_classes]
    preds = logits.argmax(dim=1)
    probs = F.softmax(logits, dim=1)[:, 1]  # Prob for class 1
    return preds, probs
