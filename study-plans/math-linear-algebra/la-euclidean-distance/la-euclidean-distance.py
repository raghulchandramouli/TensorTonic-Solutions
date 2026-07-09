import numpy as np
import math 
def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.linalg.norm(x - y))