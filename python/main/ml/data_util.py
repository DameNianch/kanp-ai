from PIL import Image, ImageOps
import numpy as np


def simple_augment(image):
    """
    image: pillowのImageオブジェクト
    """
    # HACK: 関数の命名がクソ
    image = image.rotate(np.random.randint(0, 180))
    if np.random.randint(0, 100) > 49:
        image = ImageOps.flip(image)
    # なにかaugment追加
    return image


def load_images(image_paths):
    image_arr = []
    for i_path in image_paths:
        image = simple_augment(Image.open(i_path))
        image_arr.append(np.array(image) / 255)
    return np.array(image_arr)
