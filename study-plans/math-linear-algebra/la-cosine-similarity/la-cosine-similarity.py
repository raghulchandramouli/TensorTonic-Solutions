import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """

    a = np.asarray(a)
    b = np.asarray(b)

    upper = a @ b
    lower_a = np.linalg.norm(a)
    lower_b = np.linalg.norm(b)

    if lower_a < 1e-10 or lower_b < 1e-10:
        return 0.000
    return float(upper / (lower_a * lower_b))