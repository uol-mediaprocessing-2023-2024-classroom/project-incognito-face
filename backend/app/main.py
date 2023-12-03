import io
import math
import os
import ssl
from contextlib import redirect_stdout

import dlib
import base64
import urllib.request
import concurrent.futures

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from PIL import Image, ImageFilter
from mtcnn import MTCNN
from pydantic import BaseModel
from pathlib import Path
from io import BytesIO
import cv2
import numpy
from matplotlib import pyplot as plt

app = FastAPI()
IMAGE_PATH = Path(__file__).parent.parent.parent / 'images'
FACE_DETECTION_ALGORITHMS = [
    {
        'name': 'viola-jones',
        'displayName': 'Viola Jones'
    },
    {
        'name': 'hog-svm',
        'displayName': 'HOG-SVM'
    },
    # TODO: CNN needs A LOT of time to run (30sec?) and his confidence is broken (109%?)
    # {
    #    'name': 'cnn',
    #    'displayName': 'CNN'
    # },
    {
        'name': 'mtcnn',
        'displayName': 'MTCNN'
    },
    {
        'name': 'ssd',
        'displayName': 'SSD'
    }
]
FILTERS = [
    {
        'name': 'blur',
        'displayName': 'Blur'
    },
    {
        'name': 'horizontalEdge',
        'displayName': 'Horizontal Edges'
    },
    {
        'name': 'verticalEdge',
        'displayName': 'Vertical Edges'
    },
    {
        'name': 'sunglasses',
        'displayName': 'Sunglasses'
    },
    {
        'name': 'faceMask',
        'displayName': 'Face Mask'
    },
    {
        'name': 'medicineMask',
        'displayName': 'Medicine Mask'
    },
    {
        'name': 'cowFace',
        'displayName': 'Cow Face'
    },
    {
        'name': 'saltNPepper',
        'displayName': 'Salt and Pepper'
    }
]

# SSL configuration for HTTPS requests
ssl._create_default_https_context = ssl._create_unverified_context

# CORS configuration: specify the origins that are allowed to make cross-site requests
origins = [
    'https://localhost:8080',
    'http://localhost:8080',
    'http://localhost:8081',
    'https://localhost:8081'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# A simple endpoint to verify that the API is online.
@app.get('/')
def home():
    return {'Test': 'Online'}


@app.get('/get-images')
async def get_images():
    image_data_list = []
    image_files = list(IMAGE_PATH.glob('*.jpg'))
    # TODO: Add png images
    for img_path in image_files:
        image_data = get_image_data(img_path)
        image_data_list.append(image_data)
    return JSONResponse(content=image_data_list)


def get_image_data(img_path):
    with open(img_path, 'rb') as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return {
        'name': img_path.name,
        'timestamp': os.path.getmtime(img_path),
        'base64': f'data:image/png;base64,{base64_encoded}'
    }


@app.get('/get-filters')
async def get_filters():
    return JSONResponse(content=FILTERS)


@app.get('/get-algorithms')
async def get_algorithms():
    return JSONResponse(content=FACE_DETECTION_ALGORITHMS)


class FilterRequestData(BaseModel):
    filter: str
    base64: str


@app.post('/apply-filter')
async def apply_filter(data: FilterRequestData):
    img = Image.open(BytesIO(base64.b64decode(data.base64[22:])))
    match data.filter:
        case 'blur':
            img = apply_blur(img)
        case 'horizontalEdge':
            img = apply_vertical_edge(img)
        case 'verticalEdge':
            img = apply_horizontal_edge(img)
        case 'sunglasses':
            img = apply_sunglasses(img)
        case 'faceMask':
            img = apply_whole_face_mask(img)
        case 'medicineMask':
            img = apply_medicine_mask(img)
        case 'cowFace':
            img = apply_cow_pattern(img)
        case 'saltNPepper':
            img = apply_salt_n_pepper(img)

    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    return JSONResponse(
        content={'base64': f'data:image/png;base64,{base64.b64encode(img_bytes_io.getvalue()).decode("utf-8")}'})


class RunFaceDetectionRequestData(BaseModel):
    base64: str


@app.post('/run-face-detection')
async def get_face_data(data: RunFaceDetectionRequestData):
    result = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_algorithm, algorithm, data.base64) for algorithm in
                   FACE_DETECTION_ALGORITHMS]
        for future in concurrent.futures.as_completed(futures):
            try:
                result.append(future.result())
            except Exception as e:
                print(f'An error occurred while trying to run multi-threaded face detection: {e}')
    return JSONResponse(content=result)


class GetBoxAndKeypointsRequestData(BaseModel):
    base64: str


# TODO: Doesnt work - Daten sollten entweder auf dem Server gespeichert werden oder sie müssen bei jedem Filter immer wieder zum Server geschickt werden
@app.post('/get-box-and-keypoints')
async def get_box_and_keypoints(data: GetBoxAndKeypointsRequestData) -> list[tuple[list, dict]]:
    detector = MTCNN()

    # Disable printing
    with io.StringIO() as dummy_stdout:
        with redirect_stdout(dummy_stdout):
            detected_faces = detector.detect_faces(data.base64)

    box_and_keypoints_list = []
    for face in detected_faces:
        box_and_keypoints_list.append((face['box'], face['keypoints']))

    return JSONResponse(content=box_and_keypoints_list)


def process_algorithm(algorithm, img):
    img = Image.open(BytesIO(base64.b64decode(img[22:])))
    match algorithm['name']:
        case 'viola-jones':
            img, has_face, confidence = highlight_face_viola_jones(img)
        case 'hog-svm':
            img, has_face, confidence = highlight_face_hog_svm(img)
        case 'cnn':
            img, has_face, confidence = highlight_face_cnn(img)
        case 'mtcnn':
            img, has_face, confidence = highlight_face_mtcnn(img)
        case 'ssd':
            img, has_face, confidence = highlight_face_ssd(img)
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    return {
        'name': algorithm['name'],
        'base64': f'data:image/png;base64,{base64.b64encode(img_bytes_io.getvalue()).decode("utf-8")}',
        'has_face': has_face,
        'confidence': confidence
    }


def find_faces_with_mtcnn(image: numpy.ndarray) -> list[tuple[list, dict]]:
    # Disable printing
    with (io.StringIO() as dummy_stdout):
        with redirect_stdout(dummy_stdout):
            detected_faces = mtcnn_detector.detect_faces(image)

    box_and_keypoints_list = []
    for face in detected_faces:
        box_and_keypoints_list.append((face['box'], face['keypoints']))

    return box_and_keypoints_list


# Opens the image from the given path and applies a box blur effect.
def apply_blur(img: Image):
    img = img.filter(ImageFilter.BoxBlur(10))
    return img


def apply_vertical_edge(img: Image):
    img = img.filter(ImageFilter.Kernel((3, 3), (-1, 0, 1, -2, 0, 2, -1, 0, 1), 1, 0))
    return img


def apply_horizontal_edge(img: Image):
    img = img.filter(ImageFilter.Kernel((3, 3), (-1, -2, -1, 0, 0, 0, 1, 2, 1), 1, 0))
    return img


def apply_max_filter(img: Image):
    img = img.filter(ImageFilter.MaxFilter(3))
    return img


def apply_sunglasses(img: Image, scale_factor: float = 2.5):
    foreground = Image.open('filters/sunglasses.png')

    # TODO: Calculate list when loading the selectedImage
    box_and_keypoints_list = find_faces_with_mtcnn(numpy.asarray(img))

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

        img.paste(rotated_overlay, left_upper_paste, rotated_overlay)

    return img


# TODO: Change Position and scale of mask
def apply_whole_face_mask(img: Image):
    foreground = Image.open('filters/whole_face_mask.png').convert("RGBA")

    # TODO: Calculate list when loading the selectedImage
    box_and_keypoints_list = find_faces_with_mtcnn(numpy.asarray(img))

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

        img.paste(rotated_overlay, left_upper_paste, rotated_overlay)

    return img


def apply_medicine_mask(img: Image):
    foreground = Image.open('filters/medicine_mask.png')

    # TODO: Calculate list when loading the selectedImage
    box_and_keypoints_list = find_faces_with_mtcnn(numpy.asarray(img))

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

        img.paste(rotated_overlay, left_upper_paste, rotated_overlay)

    return img


def apply_cow_pattern(image: Image, alpha_of_cow_pattern: int = 85):
    foreground = Image.open('../backend/filters/cow_pattern.png')
    foreground.putalpha(alpha_of_cow_pattern)

    # TODO: Calculate list when loading the selectedImage
    box_and_keypoints_list = find_faces_with_mtcnn(numpy.asarray(image))

    for (box, keypoints) in box_and_keypoints_list:
        box_upper_left_x = box[0]
        box_upper_left_y = box[1]
        box_width = box[2]
        box_height = box[3]
        resized_foreground = foreground.resize((box_width, box_height), resample=Image.LANCZOS)
        image.paste(resized_foreground, (box_upper_left_x, box_upper_left_y), resized_foreground)

    return image


def apply_salt_n_pepper(image: Image, alpha_of_salt_n_pepper: int = 90):
    # TODO: Calculate list when loading the selectedImage
    box_and_keypoints_list = find_faces_with_mtcnn(numpy.asarray(image))

    for (box, keypoints) in box_and_keypoints_list:
        box_upper_left_x = box[0]
        box_upper_left_y = box[1]
        box_width = box[2]
        box_height = box[3]
        pixels = numpy.zeros(box_width * box_height, dtype=numpy.uint8)
        pixels[:box_width * box_height // 2] = 255  # Set first half to white (value 255)
        numpy.random.shuffle(pixels)
        rgb_box = numpy.stack((pixels, pixels, pixels), axis=-1)
        rgb_box_reshaped = numpy.reshape(rgb_box, (box_height, box_width, 3))
        pixels_with_alpha = Image.fromarray(rgb_box_reshaped)
        pixels_with_alpha.putalpha(alpha_of_salt_n_pepper)
        image.paste(pixels_with_alpha, (box_upper_left_x, box_upper_left_y), pixels_with_alpha)

    return image

def highlight_face_viola_jones(img: Image):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = viola_jones_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    detected_a_face = False

    for (x, y, w, h) in face:
        detected_a_face = True
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 6)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), detected_a_face, '?'


def highlight_face_hog_svm(img: Image):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = hog_svm_detector(gray_image)

    detected_a_face = False

    for face in faces:
        detected_a_face = True
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 6)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), detected_a_face, '?'


def highlight_face_cnn(img: Image):
    img = numpy.array(img)
    faces = cnn_detector(img)

    detected_a_face = False
    confidence = 100

    for face in faces:
        detected_a_face = True
        confidence = round(face.confidence * 100, 3)
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 6)

    return Image.fromarray(img), detected_a_face, confidence


def highlight_face_mtcnn(img: Image):
    img = numpy.array(img)
    # Disable printing
    with io.StringIO() as dummy_stdout:
        with redirect_stdout(dummy_stdout):
            faces = mtcnn_detector.detect_faces(img)

    detected_a_face = False
    confidence = 100

    for face in faces:
        detected_a_face = True
        confidence = round(face['confidence'] * 100, 3)
        x, y, w, h = face['box'][0], face['box'][1], face['box'][2], face['box'][3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 6)

    return Image.fromarray(img), detected_a_face, confidence


def highlight_face_ssd(img: Image):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    resized_rgb_image = cv2.resize(img, (300, 300))
    imageBlob = cv2.dnn.blobFromImage(image=resized_rgb_image)
    ssd_detector.setInput(imageBlob)
    detections = ssd_detector.forward()

    detected_a_face = False
    confidence = 100

    # only show detections over 80% certainty
    for row in detections[0][0]:
        if row[2] > 0.80:
            detected_a_face = True
            confidence = round(row[2] * 100, 3)
            x1, y1, x2, y2 = int(row[3] * img.shape[1]), int(row[4] * img.shape[0]), int(row[5] * img.shape[1]), int(
                row[6] * img.shape[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 6)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), detected_a_face, confidence


# Global exception handler that catches all exceptions not handled by specific exception handlers.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={'message': 'An unexpected error occurred.'},
    )


# Load detectors
viola_jones_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hog_svm_detector = dlib.get_frontal_face_detector()
cnn_detector = dlib.cnn_face_detection_model_v1("resources/mmod_human_face_detector.dat")
mtcnn_detector = MTCNN()
ssd_detector = cv2.dnn.readNetFromCaffe("resources/deploy.prototxt",
                                        "resources/res10_300x300_ssd_iter_140000.caffemodel")
