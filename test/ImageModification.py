import cv2
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


def create_add_black_squares(square_size: int, number_of_squares: int):
    def add_black_squares(image: np.ndarray):
        for _ in range(number_of_squares):
            x = np.random.randint(0, image.shape[0] - square_size)
            y = np.random.randint(0, image.shape[1] - square_size)
            image[x:x + square_size, y:y + square_size] = 0  # 0 because squares are black

        # Ensure the image values stay within 0-255 range
        return np.clip(image, 0, 255).astype(np.uint8)

    return add_black_squares


def apply_bilateral_filter(image):
    # bilateral filter (explain params)
    filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    return filtered_image


def identity(image: np.ndarray):
    return image
