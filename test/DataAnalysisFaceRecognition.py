import cv2
import dlib
import face_recognition
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import datasets

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


face_pairs = get_face_pairs()
correct_found_pairs = 0
print(len(face_pairs))

for pair in face_pairs:
    pair0 = Image.fromarray((pair[0] * 255).astype(np.uint8))
    pair1 = Image.fromarray((pair[1] * 255).astype(np.uint8))
    if recognize_faces(pair0, pair1) == 1:
        correct_found_pairs += 1

print(str(correct_found_pairs) + '/' + str(len(face_pairs)))




