import io
import math
from contextlib import redirect_stdout

import cv2
import numpy
import numpy as np
import dlib
from PIL import Image as PILImage
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN


def find_faces_with_mtcnn(image: np.ndarray) -> list[tuple[list, dict]]:
    detector = MTCNN()

    # Disable printing
    with io.StringIO() as dummy_stdout:
        with redirect_stdout(dummy_stdout):
            detected_faces = detector.detect_faces(image)

    img = numpy.asarray(image)

    box_and_keypoints_list = []
    for face in detected_faces:
        box_and_keypoints_list.append((face['box'], face['keypoints']))
        x, y, w, h = face['box'][0], face['box'][1], face['box'][2], face['box'][3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 6)

    plt.imshow(img)
    plt.show()

    return box_and_keypoints_list


def apply_sunglasses(image: PILImage.Image, box_and_keypoints_list: list[tuple[list, dict]], scale_factor: float = 2.5):
    foreground = PILImage.open('../images/sunglasses.png')

    for (box, keypoints) in box_and_keypoints_list:
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_radians = math.atan2(-dy, dx)
        angle_degrees = math.degrees(angle_radians)
        eye_distance = math.dist((right_eye[0], right_eye[1]), (left_eye[0], left_eye[1]))

        foreground_width_to_height_ratio = foreground.size[0] / foreground.size[1]
        foreground = foreground.resize(size=(
            int(scale_factor * eye_distance), int(scale_factor * eye_distance / foreground_width_to_height_ratio)))

        rotated_overlay = foreground.rotate(angle_degrees, expand=True)

        left_part = (scale_factor - 1) / 2
        left_upper_sunglasses = (int(left_eye[0] - eye_distance * left_part),
                                 int(left_eye[1] - eye_distance * left_part / foreground_width_to_height_ratio))

        left_upper_paste = (left_upper_sunglasses[0], int(left_upper_sunglasses[1] - math.fabs(
            math.cos(math.radians(90 - angle_degrees)) * scale_factor * eye_distance)))

        image.paste(rotated_overlay, left_upper_paste, rotated_overlay)


# das ist nur zum spielen
def apply_whole_face_mask(image: PILImage.Image, box_and_keypoints_list: list[tuple[list, dict]]):
    foreground = PILImage.open('../images/whole_face_mask.png').convert("RGBA")

    for (box, keypoints) in box_and_keypoints_list:
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_radians = math.atan2(-dy, dx)
        angle_degrees = math.degrees(angle_radians)
        face_width = box[2]

        foreground_width_to_height_ratio = foreground.size[0] / foreground.size[1]
        foreground = foreground.resize(size=(face_width, int(face_width / foreground_width_to_height_ratio)))

        rotated_overlay = foreground.rotate(angle_degrees, expand=True)

        left_upper_face_mask = (box[0], box[1])

        left_upper_paste = (left_upper_face_mask[0], int(left_upper_face_mask[1] - math.fabs(
            math.cos(math.radians(90 - angle_degrees)) * face_width)))

        image.paste(rotated_overlay, left_upper_paste, rotated_overlay)


def apply_medicine_mask(image: PILImage.Image, box_and_keypoints_list: list[tuple[list, dict]]):
    foreground = PILImage.open('../images/medicine_mask.png')
    numpyArr = numpy.asarray(image)

    for (box, keypoints) in box_and_keypoints_list:
        left_mouth = keypoints['mouth_left']
        right_mouth = keypoints['mouth_right']
        dx = right_mouth[0] - left_mouth[0]
        dy = right_mouth[1] - left_mouth[1]
        angle_radians = math.atan2(-dy, dx)
        angle_degrees = math.degrees(angle_radians)
        face_width = box[2]

        foreground_width_to_height_ratio = foreground.size[0] / foreground.size[1]
        foreground = foreground.resize(size=(face_width, int(face_width / foreground_width_to_height_ratio)))

        rotated_overlay = foreground.rotate(angle_degrees, expand=True)

        left_upper_face_mask = (box[0], keypoints['nose'][1])

        left_upper_paste = (left_upper_face_mask[0], int(left_upper_face_mask[1] - math.fabs(
            math.cos(math.radians(90 - angle_degrees)) * face_width)))

        image.paste(rotated_overlay, left_upper_paste, rotated_overlay)

        cv2.circle(numpyArr, (keypoints['left_eye']), 2, (0, 255, 0), 2)
        cv2.circle(numpyArr, (keypoints['right_eye']), 2, (0, 255, 0), 2)
        cv2.circle(numpyArr, (keypoints['nose']), 2, (0, 255, 0), 2)
        cv2.circle(numpyArr, (keypoints['mouth_left']), 2, (0, 255, 0), 2)
        cv2.circle(numpyArr, (keypoints['mouth_right']), 2, (0, 255, 0), 2)

    plt.imshow(numpyArr)
    plt.show()


pilImage = PILImage.open('../images/politician.jpg')
box_and_keypoints_list = find_faces_with_mtcnn(np.array(pilImage))
apply_medicine_mask(pilImage, box_and_keypoints_list)
apply_sunglasses(pilImage, box_and_keypoints_list)
# apply_sunglasses(PILImage.open('../images/olaf.jpg'))
# apply_sunglasses(PILImage.open('../images/politician.jpg'))


img = cv2.cvtColor(numpy.array(pilImage), cv2.COLOR_RGB2BGR)
detector = dlib.get_frontal_face_detector()
faces = detector(img)
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 6)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

# plt.imshow(pilImage)
# plt.show()
