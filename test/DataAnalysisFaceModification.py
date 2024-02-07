import io
import math
import random
from contextlib import redirect_stdout
from typing import Callable
import os
import cv2
import numpy as np
import dlib
from sklearn import datasets
from matplotlib import pyplot as plt
import ImageModification as imod
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageFilter, ImageDraw
import numpy
import sys
import logging

sys.path.append('../backend/app')
# we always call serverfrom here.
os.chdir('../backend/')
import main as server


# this is relative based on new directory from change before!!!!
# os.chdir('../test')
logging.basicConfig(level=logging.INFO)

## Counting


def get_faces():
    return datasets.fetch_lfw_people(color=True, min_faces_per_person=100, resize=1)


def find_keypoints(image: Image):
    gray_image = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_BGR2GRAY)
    faces = server.hog_svm_detector(gray_image)
    box_and_keypoints_list = []
    for face in faces:
        all_landmarks = server.hog_svm_shape_predictor(gray_image, face)
        box = [face.left(), face.top(), face.width(), face.height()]
        face_keypoints = server.calculate_face_keypoints(all_landmarks)
        face_shape_landmarks = server.calculate_face_shape_landmarks(all_landmarks)
        face_encoding = server.calculate_face_encoding(np.asarray(image), box)
        box_and_keypoints_list.append((box, face_keypoints, face_shape_landmarks, face_encoding))

    return box_and_keypoints_list


def count_detected_faces_hog(image_mod_fn, needs_keypoints=True, **kwargs):
    faces = get_faces()
    detector = dlib.get_frontal_face_detector()
    counter_faces = 0
    logging.info(f'kwargs: {kwargs}')

    for possibleFace in faces.images:
        scaled_rgb_image = (possibleFace * 255).astype(np.uint8)

        pil_img = Image.fromarray(scaled_rgb_image)

        keypoints = find_keypoints(pil_img)

        if not needs_keypoints:
            modifiedImage = image_mod_fn(pil_img)
        elif len(kwargs) == 0:
            modifiedImage = image_mod_fn(pil_img, keypoints)
        else:
            modifiedImage = image_mod_fn(pil_img, keypoints, *kwargs.values())

        modifiedImage = np.array(modifiedImage)

        detected_faces = detector(modifiedImage)

        if len(detected_faces) > 0:
            counter_faces += 1

    print("From " + str(len(faces.images)) + " Faces the HOG algorithm detected " + str(counter_faces) + " Faces")
    return float(counter_faces) / float(len(faces.images))


def plot_before_and_after_modification(no_modification: Image, after_modification: Image, title_modification_img: str):
    fig = plt.figure(figsize=(10, 5))
    # Plotting 'before' image
    plt.subplot(1, 2, 1)
    plt.title('Unmodified Image')
    plt.imshow(no_modification)
    plt.axis('off')

    # Plotting 'after' image
    plt.subplot(1, 2, 2)
    plt.title(title_modification_img)
    plt.imshow(after_modification)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('../documentation/images/' + title_modification_img.replace(" ", ""), bbox_inches='tight')
    plt.close()


# Image Modification Functions only have potential alpha values as parameters,
# so this function only checks for alpha values. Refactor this once it changes (not worth the time rn)
def image_modification_plot_hog_and_alpha(title: str, modification_description: str, image_mod_fn, alpha_values=None):
    categories = []
    values = []

    if alpha_values is None:
        categories = ["Unmodified Image", "Image with" + modification_description]
        values = [count_detected_faces_hog(lambda x: x, False), count_detected_faces_hog(image_mod_fn)]
    else:
        categories.append("Unmodified Image")
        values.append(count_detected_faces_hog(lambda x: x, False))
        for alpha_value in alpha_values:
            categories.append(modification_description + ' with alpha of ' + str(alpha_value))
            values.append(count_detected_faces_hog(image_mod_fn, needs_keypoints = True, parameters = alpha_value))

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
    plt.savefig('../documentation/images/' + title.replace(" ", ""), bbox_inches='tight')
    plt.close()


#Setup
#plot_img = get_faces().images[0]
#scaled_plot = (plot_img * 255).astype(np.uint8)
#pil_img = Image.fromarray(scaled_plot)
#pil_img_copy = pil_img.copy()
#keypoints_plot_img = find_keypoints(pil_img)

# Cowface

# plot_before_and_after_modification(pil_img_copy,
#                                    server.apply_cow_pattern(pil_img, keypoints_plot_img, alpha_of_cow_pattern=80),
#                                    'Cow Mask with Alpha of 45')
#image_modification_plot_hog_and_alpha('Cow Mask Modification', 'Cow Mask', server.apply_cow_pattern,
#                                       alpha_values = [50, 100, 150, 200, 250])

# Salt and Pepper
# plot_before_and_after_modification(pil_img_copy,
#                                    server.apply_salt_n_pepper(pil_img, keypoints_plot_img, alpha_of_salt_n_pepper=45),
#                                    'Salt and Pepper with alpha of 45')
#image_modification_plot_hog_and_alpha('Salt and Pepper Modification', 'Salt and Pepper', server.apply_salt_n_pepper,
#                                       alpha_values = [50, 100, 150, 200, 250])

# Sunglasses
# plot_before_and_after_modification(pil_img_copy,
#                                    server.apply_sunglasses(pil_img, keypoints_plot_img),
#                                    'Sunglasses on Face')
# image_modification_plot_hog_and_alpha('Sunglasses on Face Modification', ' Applied Sunglasses', server.apply_sunglasses,
#                                     )

# Medicine Mask
# plot_before_and_after_modification(pil_img_copy,
#                                    server.apply_medicine_mask(pil_img, keypoints_plot_img),
#                                    'Medicine Mask Example')
# image_modification_plot_hog_and_alpha('Medicine Mask Modification', ' Applied Medicine Mask', server.apply_medicine_mask,
#                                      )

# Hide with Face
# plot_before_and_after_modification(pil_img_copy,
#                                    server.apply_medicine_mask(pil_img, keypoints_plot_img),
#                                    'Hide With Face')

# Pixelate
# plot_before_and_after_modification(pil_img_copy, server.apply_pixelate(pil_img, keypoints_plot_img, pixel_size=2), 'Pixelate')
#image_modification_plot_hog_and_alpha('Pixelate Modification', 'Applied Pixelate', server.apply_pixelate, alpha_values = [2,4,6,8])

# Blur
#plot_before_and_after_modification(pil_img_copy, server.apply_blur(pil_img, keypoints_plot_img), 'Box Blur')
#image_modification_plot_hog_and_alpha('Box Blur Modification', ' Box Blur with radius of 1', server.apply_blur)

