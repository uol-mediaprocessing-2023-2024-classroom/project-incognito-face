from typing import Callable

import cv2
import numpy as np
import dlib
from PIL import Image, ImageFilter
from sklearn import datasets
from matplotlib import pyplot as plt


def get_faces():
    return datasets.fetch_lfw_people(color=True, min_faces_per_person=100, resize=1)


def count_detected_faces_haar(image_mod_fn: Callable[[np.ndarray], np.ndarray]):
    faces = get_faces()
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    counter_faces = 0
    plt.matshow(faces.images[0])
    plt.show()

    for possibleFace in faces.images:

        scaled_rgb_image = (possibleFace * 255).astype(np.uint8)

        modifiedImage = image_mod_fn(scaled_rgb_image)

        detected_faces = face_classifier.detectMultiScale(modifiedImage, scaleFactor=1.1, minNeighbors=5,
                                                          minSize=(40, 40))

        if len(detected_faces) > 0:
            counter_faces += 1

    print("From " + str(len(faces.images)) + " Faces the Haar algorithm detected " + str(counter_faces) + " Faces")


def count_detected_faces_hog(image_mod_fn: Callable[[np.ndarray], np.ndarray]):
    faces = get_faces()
    detector = dlib.get_frontal_face_detector()
    counter_faces = 0
    plt.matshow(faces.images[0])
    plt.show()

    for possibleFace in faces.images:

        scaled_rgb_image = (possibleFace * 255).astype(np.uint8)

        modifiedImage = image_mod_fn(scaled_rgb_image)

        detected_faces = detector(modifiedImage)

        if len(detected_faces) > 0:
            counter_faces += 1

    print("From " + str(len(faces.images)) + " Faces the HOG algorithm detected " + str(counter_faces) + " Faces")


def apply_blur(image: np.ndarray):
    pilImage = Image.fromarray(image)
    pilImage = pilImage.filter(ImageFilter.BoxBlur(10))
    return np.array(pilImage)


def rotate_image(image: np.ndarray):
    pilImage = Image.fromarray(image)
    pilImage = pilImage.rotate(angle=10)
    return np.array(pilImage)


count_detected_faces_haar(rotate_image)
count_detected_faces_hog(rotate_image)
