import io
from contextlib import redirect_stdout
from typing import Callable

import cv2
import numpy as np
import dlib
from PIL import Image, ImageFilter, ImageOps
from sklearn import datasets
from matplotlib import pyplot as plt
import ImageModification as imod
from mtcnn.mtcnn import MTCNN

def get_faces():
    return datasets.fetch_lfw_people(color=True, min_faces_per_person=100, resize=1)


def count_detected_faces_haar(image_mod_fn: Callable[[np.ndarray], np.ndarray]):
    faces = get_faces()
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    counter_faces = 0

    for possibleFace in faces.images:

        scaled_rgb_image = (possibleFace * 255).astype(np.uint8)

        modifiedImage = image_mod_fn(scaled_rgb_image)

        detected_faces = face_classifier.detectMultiScale(modifiedImage, scaleFactor=1.1, minNeighbors=5,
                                                          minSize=(40, 40))

        if len(detected_faces) > 0:
            counter_faces += 1

    print("From " + str(len(faces.images)) + " Faces the Haar algorithm detected " + str(counter_faces) + " Faces")
    return float(counter_faces) / float(len(faces.images))


def count_detected_faces_hog(image_mod_fn: Callable[[np.ndarray], np.ndarray]):
    faces = get_faces()
    detector = dlib.get_frontal_face_detector()
    counter_faces = 0

    for possibleFace in faces.images:
        scaled_rgb_image = (possibleFace * 255).astype(np.uint8)

        modifiedImage = image_mod_fn(scaled_rgb_image)

        detected_faces = detector(modifiedImage)

        if len(detected_faces) > 0:
            counter_faces += 1

    print("From " + str(len(faces.images)) + " Faces the HOG algorithm detected " + str(counter_faces) + " Faces")
    return float(counter_faces) / float(len(faces.images))


# http://dlib.net/cnn_face_detector.py.html
def count_detected_faces_cnn(image_mod_fn: Callable[[np.ndarray], np.ndarray]):
    faces = get_faces()
    detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    counter_faces = 0

    for possibleFace in faces.images:
        scaled_rgb_image = (possibleFace * 255).astype(np.uint8)

        modifiedImage = image_mod_fn(scaled_rgb_image)

        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        # detected_faces = detector(modifiedImage, 1)
        detected_faces = detector(modifiedImage)

        if len(detected_faces) > 0:
            counter_faces += 1

    print("From " + str(len(faces.images)) + " Faces the CNN algorithm detected " + str(counter_faces) + " Faces")
    return float(counter_faces) / float(len(faces.images))


# https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/
def count_detected_faces_ssd(image_mod_fn: Callable[[np.ndarray], np.ndarray]):
    faces = get_faces()
    detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    counter_faces = 0

    for possibleFace in faces.images:
        scaled_rgb_image = (possibleFace * 255).astype(np.uint8)
        resized_scaled_rgb_image = cv2.resize(scaled_rgb_image, (300, 300))

        modifiedImage = image_mod_fn(resized_scaled_rgb_image)

        imageBlob = cv2.dnn.blobFromImage(image=modifiedImage)

        detector.setInput(imageBlob)
        detections = detector.forward()

        found_face = False
        for row in detections[0][0]:
            if row[2] > 0.90:
                found_face = True
                break

        if found_face > 0:
            counter_faces += 1

    print("From " + str(len(faces.images)) + " Faces the SSD algorithm detected " + str(counter_faces) + " Faces")
    return float(counter_faces) / float(len(faces.images))


# https://www.kaggle.com/code/jake126/face-detection-using-cnn-with-the-lfw-dataset
def count_detected_faces_mtcnn(image_mod_fn: Callable[[np.ndarray], np.ndarray]):
    faces = get_faces()
    detector = MTCNN()
    counter_faces = 0

    for possibleFace in faces.images:
        scaled_rgb_image = (possibleFace * 255).astype(np.uint8)

        modifiedImage = image_mod_fn(scaled_rgb_image)

        # Disable printing
        with io.StringIO() as dummy_stdout:
            with redirect_stdout(dummy_stdout):
                detected_faces = detector.detect_faces(modifiedImage)

        if len(detected_faces) > 0:
            counter_faces += 1

    print("From " + str(len(faces.images)) + " Faces the MTCNN algorithm detected " + str(counter_faces) + " Faces")
    return float(counter_faces) / float(len(faces.images))


def image_modification_plot(count_detected_faces: Callable[[Callable[[np.ndarray], np.ndarray]], float], title: str):
    categories = ["Unmodified", "Blur", "Rotate 20°", "Rotate 90°", "Flip Horizontally", "Change to Grayscale"]
    values = [count_detected_faces(imod.identity), count_detected_faces(imod.apply_blur),
              count_detected_faces(imod.create_rotate_image(20)),
              count_detected_faces(imod.create_rotate_image(90)), count_detected_faces(imod.flip_image_horizontally),
              count_detected_faces(imod.change_to_grayscale)]

    # Create bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(categories, values, color='skyblue')
    plt.ylim(0, 1.0)

    # Title and labels
    plt.title(title)
    plt.xlabel('Modification Operations')
    plt.ylabel('Proportion detected')

    # Show plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.show()


def find_face_with_mtcnn(image: np.ndarray):
    detector = MTCNN()

    # Disable printing
    with io.StringIO() as dummy_stdout:
        with redirect_stdout(dummy_stdout):
            detected_faces = detector.detect_faces(image)

    box_coords_faces = []
    for face in detected_faces:
        bbox = face['box']
        #keypoints = face['keypoints']

        #cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        #cv2.circle(image, (keypoints['left_eye']), 2, (0, 255, 0), 2)
        #cv2.circle(image, (keypoints['right_eye']), 2, (0, 255, 0), 2)
        #cv2.circle(image, (keypoints['nose']), 2, (0, 255, 0), 2)
        #cv2.circle(image, (keypoints['mouth_left']), 2, (0, 255, 0), 2)
        #cv2.circle(image, (keypoints['mouth_right']), 2, (0, 255, 0), 2)
        box_coords_faces.append((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

    plt.imshow(image)
    plt.show()

    return box_coords_faces


def find_face_with_hog(image: np.ndarray):
    # detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    detector = dlib.get_frontal_face_detector()

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    # detected_faces = detector(modifiedImage, 1)
    detected_faces = detector(image)

    box_coords_faces = []
    for face in detected_faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        box_coords_faces.append((x, y, x + w, y + h))
    return box_coords_faces


def apply_face_modification(image, coordinates):
    foreground = Image.open('../images/sunglasses.png')

    for coords in coordinates:
        cur_height = coords[3] - coords[1]
        coords = coords[0], coords[1] + int(1 / 5 * cur_height), coords[2], coords[3] - int(3 / 5 * cur_height)
        width = coords[2] - coords[0]
        height = coords[3] - coords[1]
        resized_foreground = foreground.resize(size=(width, height))
        print(coords[3] - coords[1])
        print(resized_foreground.size)
        image.paste(resized_foreground, coords, mask=resized_foreground)
    return image


test_face = Image.open('../images/politician.jpg')
all_coordinates = find_face_with_mtcnn(np.array(test_face))
plt.imshow(apply_face_modification(test_face, all_coordinates))
plt.show()

# image_modification_plot(count_detected_faces_hog, 'HOG modifications')

# count_detected_faces_haar(imod.blur_edges)
# count_detected_faces_hog(imod.blur_edges)

# image_modification_plot(count_detected_faces_cnn, 'CNN modifications')

# image_modification_plot(count_detected_faces_ssd, 'SSD modifications')

image_modification_plot(count_detected_faces_mtcnn, 'MTCNN modifications')


# imod.blur_edges((get_faces().images[0] * 255).astype(np.uint8))
