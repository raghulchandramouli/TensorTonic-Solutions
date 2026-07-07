import torch
import torch.nn as nn

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """

    t = torch.tensor(x, dtype=torch.float32)
    if method == "relu":
        return torch.relu(t).tolist()
    elif method == "sigmoid":
        return torch.sigmoid(t).tolist()
    elif method == "tanh":
        return torch.tanh(t).tolist()
    elif method == "leaky_relu":
        return torch.nn.functional.leaky_relu(t).tolist()

    
        