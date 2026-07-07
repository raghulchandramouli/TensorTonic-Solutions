import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """

    t = torch.tensor(values, dtype=torch.float32, requires_grad = True)
    y = (t ** 3 + 2 * t).sum()
    y.backward()
    return t.grad.tolist()
