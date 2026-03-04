import numpy as np


def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """

    grayscale = []
    gr, gg, gb = 0.299, 0.587, 0.114
    for row in image:
        new_row = []

        for pixel in row:
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]

            gray_values = gr * r + g * gg + gb * b
            new_row.append(gray_values)
        grayscale.append(new_row)
    return grayscale