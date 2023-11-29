import io
import math
from contextlib import redirect_stdout
from typing import Callable

import cv2
import numpy as np
import dlib
from PIL import Image as PILImage
from PIL import ImageFilter, ImageOps
from sklearn import datasets
from matplotlib import pyplot as plt
import ImageModification as imod
from mtcnn.mtcnn import MTCNN
from typing import List, Tuple
import DataAnalysis


def find_faces_with_mtcnn(image: np.ndarray) -> list[tuple[list, dict]]:
    detector = MTCNN()

    # Disable printing
    with io.StringIO() as dummy_stdout:
        with redirect_stdout(dummy_stdout):
            detected_faces = detector.detect_faces(image)

    box_and_keypoints_list = []
    for face in detected_faces:
        box_and_keypoints_list.append((face['box'], face['keypoints']))

    return box_and_keypoints_list


def apply_sunglasses(image: PILImage.Image):
    foreground = PILImage.open('../images/sunglasses.png')
    box_and_keypoints_list = find_faces_with_mtcnn(np.array(image))
    for (box, keypoints) in box_and_keypoints_list:
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        rotated_overlay = foreground.rotate(angle_degrees, expand=True)
        image.paste(rotated_overlay, (left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]), rotated_overlay)

    plt.imshow(image)
    plt.show()

    # for coords in box_and_keypoints_list:
    #     cur_height = coords[3] - coords[1]
    #     coords = coords[0], coords[1] + int(1 / 5 * cur_height), coords[2], coords[3] - int(3 / 5 * cur_height)
    #     width = coords[2] - coords[0]
    #     height = coords[3] - coords[1]
    #     resized_foreground = foreground.resize(size=(width, height))
    #     print(coords[3] - coords[1])
    #     print(resized_foreground.size)
    #     image.paste(resized_foreground, coords, mask=resized_foreground)
    # return image



apply_sunglasses(PILImage.open('../images/christian.jpg'))
