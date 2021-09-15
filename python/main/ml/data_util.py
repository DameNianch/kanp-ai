from PIL import Image
import numpy as np


def load_images(image_paths):
    image_arr = []
    for i_path in image_paths:
        image_arr.append(np.array(Image.open(i_path)))
    return np.array(image_arr)
