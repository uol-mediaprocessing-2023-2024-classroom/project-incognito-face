import io
import math
from contextlib import redirect_stdout

import cv2
import numpy
import numpy as np
import dlib
from PIL import Image as PILImage, Image
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
    foreground = PILImage.open('../backend/filters/sunglasses.png')

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
    foreground = PILImage.open('../backend/filters/whole_face_mask.png').convert("RGBA")

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
    foreground = PILImage.open('../backend/filters/medicine_mask.png')
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


# pilImage = PILImage.open('../images/politician.jpg')
# box_and_keypoints_list = find_faces_with_mtcnn(np.array(pilImage))
# apply_medicine_mask(pilImage, box_and_keypoints_list)
# apply_sunglasses(pilImage, box_and_keypoints_list)
# # apply_sunglasses(PILImage.open('../images/olaf.jpg'))
# # apply_sunglasses(PILImage.open('../images/politician.jpg'))
#
#
# img = cv2.cvtColor(numpy.array(pilImage), cv2.COLOR_RGB2BGR)
# detector = dlib.get_frontal_face_detector()
# faces = detector(img)
# for face in faces:
#     x, y, w, h = face.left(), face.top(), face.width(), face.height()
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 6)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# plt.show()

# plt.imshow(pilImage)
# plt.show()

ssd_detector = cv2.dnn.readNetFromCaffe("../backend/resources/deploy.prototxt",
                                        "../backend/resources/res10_300x300_ssd_iter_140000.caffemodel")


def highlight_face_ssd(img: Image):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    resized_rgb_image = cv2.resize(img, (300, 300))
    imageBlob = cv2.dnn.blobFromImage(image=resized_rgb_image)
    ssd_detector.setInput(imageBlob)
    detections = ssd_detector.forward()

    # only show detections over 80% certainty
    for row in detections[0][0]:
        if row[2] > 0.80:
            x1, y1, x2, y2 = int(row[3] * img.shape[1]), int(row[4] * img.shape[0]), int(row[5] * img.shape[1]), int(row[6] * img.shape[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 6)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), False, 50


pilImage = PILImage.open('../images/politician.jpg')
pilImage, has_face, conf = highlight_face_ssd(pilImage)

plt.imshow(pilImage)
plt.show()
