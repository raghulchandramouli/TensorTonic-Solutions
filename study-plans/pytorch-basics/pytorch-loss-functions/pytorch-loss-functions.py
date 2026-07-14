import torch
import torch.nn as nn

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """

    pred = torch.tensor(pred)

    if method == "cross_entropy":
        pred = pred.float()
        target = torch.tensor(target, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

    else:
        pred = pred.float()
        target = torch.tensor(target, dtype=torch.float32)

        if method == "mse":
            criterion = nn.MSELoss()

        elif method == "huber":
            criterion = nn.HuberLoss(delta=delta)

        else:
            raise ValueError(f"Unknown loss method: {method}")

    loss = criterion(pred, target)
    return loss.item()