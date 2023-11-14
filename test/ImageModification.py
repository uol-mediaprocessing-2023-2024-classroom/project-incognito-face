import numpy as np
from PIL import Image, ImageFilter


def apply_blur(image: np.ndarray):
    pilImage = Image.fromarray(image)
    pilImage = pilImage.filter(ImageFilter.BoxBlur(10))
    return np.array(pilImage)


def create_rotate_image(angle: int):
    def rotate_image(image: np.ndarray):
        pilImage = Image.fromarray(image)
        pilImage = pilImage.rotate(angle=angle)
        return np.array(pilImage)

    return rotate_image


def flip_image_horizontally(image: np.ndarray):
    pilImage = Image.fromarray(image)
    pilImage = pilImage.transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(pilImage)


def change_to_grayscale(image: np.ndarray):
    pilImage = Image.fromarray(image)
    pilImage = pilImage.convert('L')
    return np.array(pilImage)


def identity(image: np.ndarray):
    return image
