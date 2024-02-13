import os
import sys

import cv2
import dlib
import face_recognition
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import datasets

sys.path.append('../backend/app')
# we always call serverfrom here.
os.chdir('../backend/')
import main as server

hog_svm_detector = dlib.get_frontal_face_detector()


def recognize_faces(orig_img: Image, mod_img: Image):
    orig_face_encodings = calculate_face_encodings(orig_img)
    mod_face_encodings = calculate_face_encodings(mod_img)

    count_of_matches = 0

    if len(orig_face_encodings) == 0 or len(mod_face_encodings) == 0:
        return 0

    for j, face_encoding_unknown in enumerate(mod_face_encodings):
        matches = face_recognition.compare_faces(orig_face_encodings, face_encoding_unknown, tolerance=0.55)

        for i, match in enumerate(matches):
            if match:
                count_of_matches += 1

    return count_of_matches


def calculate_face_encodings(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(np.asarray(img_bgr), cv2.COLOR_BGR2GRAY)
    faces = hog_svm_detector(gray_image)
    face_encodings = []
    for face in faces:
        box = [face.left(), face.top(), face.width(), face.height()]
        face_encodings.append(calculate_face_encoding(np.asarray(img_bgr), box))

    return face_encodings


def calculate_face_encoding(image, box):
    converted_box = [(box[1], box[0] + box[2], box[1] + box[3], box[0])]
    return face_recognition.face_encodings(image, converted_box, model="large")[0]


def get_face_pairs():
    lfw_pairs = datasets.fetch_lfw_pairs(color=True, resize=1)
    X_pairs = lfw_pairs.pairs
    y_pairs = lfw_pairs.target
    return X_pairs[y_pairs == 1]


def find_keypoints(image: Image):
    gray_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
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


def run_pairs_with_filter(face_pairs):
    correct_found_pairs = 0
    for pair in face_pairs:
        if recognize_faces(pair[0], pair[1]) == 1:
            correct_found_pairs += 1

    print(str(correct_found_pairs) + '/' + str(len(face_pairs)))
    return correct_found_pairs


def plot_results(title: str, categories, values):

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


def run_analysis():
    face_pairs = get_face_pairs()
    face_pairs_pil_with_keypoints = []

    # face_pairs = face_pairs[1:5]

    for pair in face_pairs:
        pil_0 = Image.fromarray((pair[0] * 255).astype(np.uint8))
        pil_1 = Image.fromarray((pair[1] * 255).astype(np.uint8))
        keypoints = find_keypoints(pil_1)
        face_pairs_pil_with_keypoints.append((pil_0, pil_1, keypoints))

    print("Calculated Keypoints")

    cur_pairs = []
    for pair in face_pairs_pil_with_keypoints:
        cur_pairs.append((pair[0], pair[1].copy()))
    unmodified_result = run_pairs_with_filter(cur_pairs) / len(face_pairs_pil_with_keypoints)
    print("Finished match calculation")

    plt.imshow(cur_pairs[0][0])
    plt.show()
    plt.imshow(cur_pairs[0][1])
    plt.show()

    cur_pairs = []
    for pair in face_pairs_pil_with_keypoints:
        cur_pairs.append((pair[0], server.apply_cow_pattern(pair[1].copy(), pair[2])))
    cow_pattern_result = run_pairs_with_filter(cur_pairs) / len(face_pairs_pil_with_keypoints)
    print("Finished match calculation")

    plt.imshow(cur_pairs[0][1])
    plt.show()

    cur_pairs = []
    for pair in face_pairs_pil_with_keypoints:
        cur_pairs.append((pair[0], server.apply_salt_n_pepper(pair[1].copy(), pair[2], alpha_of_salt_n_pepper=75)))
    salt_and_pepper_result = run_pairs_with_filter(cur_pairs) / len(face_pairs_pil_with_keypoints)
    print("Finished match calculation")

    plt.imshow(cur_pairs[0][1])
    plt.show()

    cur_pairs = []
    for pair in face_pairs_pil_with_keypoints:
        cur_pairs.append((pair[0], server.apply_pixelate(pair[1].copy(), pair[2], pixel_size=5)))
    pixelate_result = run_pairs_with_filter(cur_pairs) / len(face_pairs_pil_with_keypoints)
    print("Finished match calculation")

    plt.imshow(cur_pairs[0][1])
    plt.show()

    cur_pairs = []
    for pair in face_pairs_pil_with_keypoints:
        cur_pairs.append((pair[0], server.apply_blur(pair[1].copy(), pair[2], strength=5)))
    blur_result = run_pairs_with_filter(cur_pairs) / len(face_pairs_pil_with_keypoints)
    print("Finished match calculation")

    plt.imshow(cur_pairs[0][1])
    plt.show()

    cur_pairs = []
    for pair in face_pairs_pil_with_keypoints:
        cur_pairs.append((pair[0], server.apply_sunglasses(pair[1].copy(), pair[2])))
    sunglasses_result = run_pairs_with_filter(cur_pairs) / len(face_pairs_pil_with_keypoints)
    print("Finished match calculation")

    plt.imshow(cur_pairs[0][1])
    plt.show()

    cur_pairs = []
    for pair in face_pairs_pil_with_keypoints:
        cur_pairs.append((pair[0], server.apply_medicine_mask(pair[1].copy(), pair[2])))
    medicine_mask_result = run_pairs_with_filter(cur_pairs) / len(face_pairs_pil_with_keypoints)
    print("Finished match calculation")

    plt.imshow(cur_pairs[0][1])
    plt.show()

    cur_pairs = []
    for pair in face_pairs_pil_with_keypoints:
        cur_pairs.append((pair[0], server.apply_morph_eyes(pair[1].copy(), pair[2])))
    morph_eyes_result = run_pairs_with_filter(cur_pairs) / len(face_pairs_pil_with_keypoints)
    print("Finished match calculation")

    plt.imshow(cur_pairs[0][1])
    plt.show()

    cur_pairs = []
    for pair in face_pairs_pil_with_keypoints:
        cur_pairs.append((pair[0], server.apply_morph_mouth(pair[1].copy(), pair[2])))
    morph_mouth_result = run_pairs_with_filter(cur_pairs) / len(face_pairs_pil_with_keypoints)
    print("Finished match calculation")

    plt.imshow(cur_pairs[0][1])
    plt.show()

    categories = ["Unmodified Image", "Cow Pattern", "Salt_&_Pepper", "Pixelate", "Blur", "Sunglasses", "Medicine Mask", "Morph Eyes", "Morph Mouth"]
    values = [unmodified_result, cow_pattern_result, salt_and_pepper_result, pixelate_result, blur_result, sunglasses_result, medicine_mask_result, morph_eyes_result, morph_mouth_result]

    plot_results("Face Recognition Analysis", categories, values)

run_analysis()

# 918/1100
# 10/1100
# 15/1100
# 183/1100
# 198/1100
# 486/1100
# 22/1100
# 11/1100
# 339/1100
