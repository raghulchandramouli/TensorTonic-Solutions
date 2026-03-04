import numpy as np


def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    image = np.array(image)
    weights = np.array([0.299, 0.587, 0.114])
    gray = np.dot(image[..., : 3], weights)
    return gray.tolist()
    