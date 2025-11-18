import numpy as np

def swap_colors(img, color_a, color_b):
    """
    img: numpy.ndarray of shape (H, W, 3), dtype=np.uint8 or compatible
    color_a: list or array, e.g. [117, 8, 240]
    color_b: list or array, e.g. [240, 128, 128]
    """
    img_copy = img.copy()
    mask_a = np.all(img_copy == color_a, axis=-1)
    mask_b = np.all(img_copy == color_b, axis=-1)

    img_copy[mask_b] = color_a
    img_copy[mask_a] = color_b 

    return img_copy