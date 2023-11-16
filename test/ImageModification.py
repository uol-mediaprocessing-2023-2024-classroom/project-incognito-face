import cv2
import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


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


def apply_gauss_noise(image: np.ndarray):
    gauss = np.random.normal(50, 10, image.shape).astype('uint8')
    return cv2.add(image, gauss)


# doesn't work right now
def blur_edges(image: np.ndarray):
    # edge detection
    grey_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    horizontalEdgeImage = convolve2d(grey_scale_image, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode='same', boundary='symm')
    verticalEdgeImage = convolve2d(grey_scale_image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode='same', boundary='symm')
    plt.matshow(horizontalEdgeImage, cmap='gray')
    plt.show()
    plt.matshow(verticalEdgeImage, cmap='gray')
    plt.show()
    edge_image = pow(pow(horizontalEdgeImage, 2) + pow(verticalEdgeImage, 2), 0.5)
    plt.matshow(edge_image, cmap='gray')
    plt.show()

    plt.matshow(np.abs(edge_image), cmap='gray')
    plt.show()

    print(edge_image.shape)
    edge_image_all_colors = np.tile(edge_image, (1, 3))

    print(edge_image_all_colors.shape)
    edge_image_all_colors_reshaped = edge_image_all_colors.reshape(edge_image.shape[0], edge_image.shape[1], 3)

    print(edge_image_all_colors_reshaped.shape)
    image_edge_blur = image + ((edge_image_all_colors_reshaped - image) * 0.2).astype('uint8')
    plt.matshow(image_edge_blur)
    plt.show()

    laplacian = cv2.Laplacian(grey_scale_image, cv2.CV_64F)
    plt.matshow(laplacian, cmap='gray')
    plt.show()

    plt.matshow(np.abs(laplacian), cmap='gray')
    plt.show()

    return image_edge_blur


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
