import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img(file_name, grayscale=False):
    try:
        color_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(file_name, color_mode)
        if img is None:
            raise ValueError(f"The file {file_name} cannot be loaded as an image.")
        return img
    except Exception as e:
        raise IOError(f"An error occurred when trying to load the image: {e}")

def display_img(image):
    if image is None:
        raise ValueError("Input image cannot be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy ndarray.")
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be a 2D (grayscale) or 3D (color) image.")
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show(block=False)