import io
import math
import os
import random
import ssl
from contextlib import redirect_stdout

import dlib
import json
import base64
import hashlib
import threading
import concurrent.futures

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from PIL import Image, ImageFilter, ImageDraw
from mtcnn import MTCNN
from pydantic import BaseModel
from pathlib import Path
from io import BytesIO
import cv2
import numpy

app = FastAPI()
IMAGE_PATH = Path(__file__).parent.parent.parent / 'images'
CACHE_FILE = IMAGE_PATH / 'cache.json'
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
        'name': 'dithering',
        'displayName': 'Floyd Steinberg'
    },
    {
        'name': 'closing',
        'displayName': 'Closing'
    },
    {
        'name': 'opening',
        'displayName': 'Opening'
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
    },
    {
        'name': 'hideWithMasks',
        'displayName': 'Hide With Masks'
    }
]

cache = {}
running_threads = {}
lock = threading.Lock()

if os.path.isfile(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'r') as file:
            cache = json.load(file)
            print(f'Loaded cache from {CACHE_FILE}')
    except:
        pass

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
    base64_encoded = f'data:image/png;base64,{base64_encoded}'
    return {
        'name': img_path.name,
        'timestamp': os.path.getmtime(img_path),
        'base64': base64_encoded,
        'hash': get_image_hash(base64_encoded)
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
    hash: str


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
        case 'dithering':
            img = apply_dithering(img)
        case 'closing':
            img = apply_closing(img)
        case 'opening':
            img = apply_opening(img)
        case 'sunglasses':
            img = apply_sunglasses(img, get_keypoints(img, True, data.hash))
        case 'faceMask':
            img = apply_whole_face_mask(img, get_keypoints(img, True, data.hash))
        case 'medicineMask':
            img = apply_medicine_mask(img, get_keypoints(img, True, data.hash))
        case 'cowFace':
            img = apply_cow_pattern(img, get_keypoints(img, True, data.hash))
        case 'saltNPepper':
            img = apply_salt_n_pepper(img, get_keypoints(img, True, data.hash))
        case 'hideWithMasks':
            img = apply_hide_with_masks(img, get_keypoints(img, True, data.hash))

    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    return JSONResponse(
        content={'base64': f'data:image/png;base64,{base64.b64encode(img_bytes_io.getvalue()).decode("utf-8")}'})


class RunFaceDetectionRequestData(BaseModel):
    base64: str
    hash: str


@app.post('/run-face-detection')
async def run_face_detection(data: RunFaceDetectionRequestData):
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


class GenerateKeypointsData(BaseModel):
    base64: str
    hash: str


@app.post('/generate-keypoints')
async def generate_keypoints(data: GenerateKeypointsData):
    image = Image.open(BytesIO(base64.b64decode(data.base64[22:])))
    get_keypoints(image, False, data.hash)
    return JSONResponse(content={'message': 'Keypoint generation started'})


def get_image_hash(image):
    img_b64 = ""
    if isinstance(image, str):
        img_b64 = image[22:]
    elif isinstance(image, Image.Image):
        img_bytes_io = BytesIO()
        image.save(img_bytes_io, format='JPEG')
        img_b64 = base64.b64encode(img_bytes_io.getvalue()).decode("utf-8")
    else:
        raise TypeError('Invalid image type')
    return hashlib.sha256(img_b64.encode()).hexdigest()


def get_keypoints(image: Image, wait_for_result=False, img_hash=None):
    if img_hash is None:
        img_hash = get_image_hash(image)
    thread = None
    with lock:
        if img_hash in cache.keys():
            return cache[img_hash]
        if img_hash in running_threads.keys():
            thread = running_threads[img_hash]
        else:
            thread = threading.Thread(target=threaded_keypoints, args=(img_hash, image))
            thread.start()
            running_threads[img_hash] = thread
    if wait_for_result:
        thread.join()
        return cache[img_hash]


def threaded_keypoints(img_hash: str, image: Image):
    gray_image = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_BGR2GRAY)
    faces = hog_svm_detector(gray_image)
    box_and_keypoints_list = []
    for face in faces:
        all_landmarks = hog_svm_shape_predictor(gray_image, face)
        box = [face.left(), face.top(), face.width(), face.height()]
        face_keypoints = calculate_face_keypoints(all_landmarks)
        face_shape_landmarks = calculate_face_shape_landmarks(all_landmarks)
        box_and_keypoints_list.append((box, face_keypoints, face_shape_landmarks))
    with lock:
        cache[img_hash] = box_and_keypoints_list
        del running_threads[img_hash]
        with open(CACHE_FILE, 'w') as file:
            json.dump(cache, file)


def calculate_face_shape_landmarks(all_landmarks):
    face_shape_landmarks = []
    for i in range(17):
        x, y = all_landmarks.part(i).x, all_landmarks.part(i).y
        face_shape_landmarks.append((x, y))

    x_mirror = (face_shape_landmarks[0][0] + face_shape_landmarks[16][0]) / 2
    y_mirror = (face_shape_landmarks[0][1] + face_shape_landmarks[16][1]) / 2
    for i in range(17):
        x, y = face_shape_landmarks[16 - i]
        face_shape_landmarks.append((x + int(2 * (x_mirror - x)), y + int(2 * (y_mirror - y))))
    return face_shape_landmarks


def calculate_face_keypoints(all_landmarks):

    # Extract landmarks for the keypoints
    left_eye_landmarks = all_landmarks.parts()[36:42]
    right_eye_landmarks = all_landmarks.parts()[42:48]

    # Calculate the midpoint of the left eye
    left_eye_midpoint = (
        sum([point.x for point in left_eye_landmarks]) // len(left_eye_landmarks),
        sum([point.y for point in left_eye_landmarks]) // len(left_eye_landmarks)
    )

    # Calculate the midpoint of the right eye
    right_eye_midpoint = (
        sum([point.x for point in right_eye_landmarks]) // len(right_eye_landmarks),
        sum([point.y for point in right_eye_landmarks]) // len(right_eye_landmarks)
    )

    # Add points to dictionary
    face_keypoints = {'left_eye': left_eye_midpoint, 'right_eye': right_eye_midpoint,
                      'left_mouth': (all_landmarks.part(48).x, all_landmarks.part(48).y),
                      'right_mouth': (all_landmarks.part(54).x, all_landmarks.part(54).y),
                      'nose': (all_landmarks.part(30).x, all_landmarks.part(30).y)}

    return face_keypoints


def process_algorithm(algorithm, img):
    img = Image.open(BytesIO(base64.b64decode(img[22:])))
    match algorithm['name']:
        case 'viola-jones':
            img, number_of_faces, confidence = highlight_face_viola_jones(img)
        case 'hog-svm':
            img, number_of_faces, confidence = highlight_face_hog_svm(img)
        case 'cnn':
            img, number_of_faces, confidence = highlight_face_cnn(img)
        case 'mtcnn':
            img, number_of_faces, confidence = highlight_face_mtcnn(img)
        case 'ssd':
            img, number_of_faces, confidence = highlight_face_ssd(img)
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    return {
        'name': algorithm['name'],
        'base64': f'data:image/png;base64,{base64.b64encode(img_bytes_io.getvalue()).decode("utf-8")}',
        'number_of_faces': number_of_faces,
        'confidence': confidence
    }


def apply_blur(image: Image):
    image = image.filter(ImageFilter.BoxBlur(10))
    return image


def apply_vertical_edge(image: Image):
    image = image.filter(ImageFilter.Kernel((3, 3), (-1, 0, 1, -2, 0, 2, -1, 0, 1), 1, 0))
    return image


def apply_horizontal_edge(image: Image):
    image = image.filter(ImageFilter.Kernel((3, 3), (-1, -2, -1, 0, 0, 0, 1, 2, 1), 1, 0))
    return image


def apply_dithering(image: Image) -> Image:
    return image.convert('RGB').quantize(colors=16, method=Image.FLOYDSTEINBERG).convert('RGB')


def apply_max_filter(image: Image):
    image = image.filter(ImageFilter.MaxFilter(9))
    return image


def apply_min_filter(image: Image):
    image = image.filter(ImageFilter.MinFilter(9))
    return image


def apply_closing(image: Image):
    return apply_min_filter(apply_max_filter(image))


def apply_opening(image: Image):
    return apply_max_filter(apply_min_filter(image))


def apply_sunglasses(image: Image, keypoints, scale_factor: float = 2.5):
    foreground = Image.open('filters/sunglasses.png')

    for (box, face_keypoints, face_shape_landmarks) in keypoints:
        left_eye = face_keypoints['left_eye']
        right_eye = face_keypoints['right_eye']
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

    return image


# TODO: Change Position and scale of mask
def apply_whole_face_mask(image: Image, keypoints):
    foreground = Image.open('filters/whole_face_mask.png').convert("RGBA")

    for (box, face_keypoints, face_shape_landmarks) in keypoints:
        left_eye = face_keypoints['left_eye']
        right_eye = face_keypoints['right_eye']
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

    return image


def apply_medicine_mask(image: Image, keypoints):
    foreground = Image.open('filters/medicine_mask.png').convert("RGBA")

    for (box, face_keypoints, face_shape_landmarks) in keypoints:
        left_mouth = face_keypoints['left_mouth']
        right_mouth = face_keypoints['right_mouth']
        dx = right_mouth[0] - left_mouth[0]
        dy = right_mouth[1] - left_mouth[1]
        angle_radians = math.atan2(-dy, dx)
        angle_degrees = math.degrees(angle_radians)
        face_width = box[2]

        foreground_width_to_height_ratio = foreground.size[0] / foreground.size[1]
        foreground = foreground.resize(size=(face_width, int(face_width / foreground_width_to_height_ratio)))

        rotated_overlay = foreground.rotate(angle_degrees, expand=True)

        left_upper_face_mask = (box[0], face_keypoints['nose'][1])

        left_upper_paste = (left_upper_face_mask[0], int(left_upper_face_mask[1] - math.fabs(
            math.cos(math.radians(90 - angle_degrees)) * face_width)))

        image.paste(rotated_overlay, left_upper_paste, rotated_overlay)

    return image


def apply_cow_pattern(image: Image, keypoints, alpha_of_cow_pattern: int = 85):
    foreground = Image.open('../backend/filters/cow_pattern.png').convert("RGBA")

    # Create foreground from filter
    foreground_parts = Image.new('RGBA', image.size)

    # Add cow pattern at face position
    for (box, face_keypoints, face_shape_landmarks) in keypoints:
        minX = min(row[0] for row in face_shape_landmarks)
        maxX = max(row[0] for row in face_shape_landmarks)
        minY = min(row[1] for row in face_shape_landmarks)
        maxY = max(row[1] for row in face_shape_landmarks)
        new_foreground_part = foreground.resize((maxX - minX, maxY - minY), resample=Image.LANCZOS)
        foreground_parts.paste(new_foreground_part, (minX, minY), new_foreground_part)

    # Apply alpha
    foreground_parts.putalpha(alpha_of_cow_pattern)

    # Create image from filters with mask
    shaped_foreground = Image.new('RGBA', image.size)

    # Create a mask image with the same size as the original image
    mask = Image.new('L', foreground_parts.size, 0)

    # Create a drawing context for the mask
    draw = ImageDraw.Draw(mask)

    # Add shape to mask
    for (box, face_keypoints, face_shape_landmarks) in keypoints:
        draw.polygon(face_shape_landmarks, fill=255)

    # Apply the mask to the new foreground
    shaped_foreground.paste(foreground_parts, mask=mask)

    # Apply the new foreground to the original image
    image.paste(shaped_foreground, mask=shaped_foreground)

    return image


def apply_salt_n_pepper(image: Image, keypoints, alpha_of_salt_n_pepper: int = 90):
    for (box, face_keypoints, face_shape_landmarks) in keypoints:
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


# ToDo Values are hardcoded change with sliders later on
def apply_hide_with_masks(img: Image, box_and_keypoints_list: list[tuple[list, dict]], number_of_masks: int = 40,
                          face_mask_width: int = 75, face_mask_height: int = 75,
                          alpha_of_masks: int = 45):
    # Apply alpha to foreground (image must have transparent background so that this works)
    foreground = Image.open('../backend/filters/whole_face_mask.png').convert('RGBA')
    foreground_alpha = apply_alpha_to_transparent_image(foreground, alpha_of_masks)

    # find coordinates of faces on image
    face_and_mask_coordinates = find_face_rectangles_mtcnn(box_and_keypoints_list)

    # find free coordinates for mask
    mask_cords = find_free_coordinates_outside_of_rectangles(img, number_of_masks, face_mask_width, face_mask_height,
                                                             face_and_mask_coordinates)

    # insert masks on image
    for mask_coords in mask_cords:
        resized_foreground = foreground_alpha.resize((face_mask_width, face_mask_height), resample=Image.LANCZOS)
        img.paste(resized_foreground, (mask_coords[0], mask_coords[1]), resized_foreground)

    return img



def highlight_face_viola_jones(img: Image):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = viola_jones_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 6)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), len(faces), '?'


def highlight_face_hog_svm(img: Image):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = hog_svm_detector(gray_image)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 6)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), len(faces), '?'


def highlight_face_cnn(img: Image):
    img = numpy.array(img)
    faces = cnn_detector(img)
    confidence = 100

    for face in faces:
        confidence = round(face.confidence * 100, 3)
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 6)

    return Image.fromarray(img), len(faces), confidence


def highlight_face_mtcnn(img: Image):
    img = numpy.array(img)
    # Disable printing
    with io.StringIO() as dummy_stdout:
        with redirect_stdout(dummy_stdout):
            faces = mtcnn_detector.detect_faces(img)

    confidence = 100

    for face in faces:
        confidence = round(face['confidence'] * 100, 3)
        x, y, w, h = face['box'][0], face['box'][1], face['box'][2], face['box'][3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 6)

    return Image.fromarray(img), len(faces), confidence


def highlight_face_ssd(img: Image):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    resized_rgb_image = cv2.resize(img, (300, 300))
    imageBlob = cv2.dnn.blobFromImage(image=resized_rgb_image)
    ssd_detector.setInput(imageBlob)
    detections = ssd_detector.forward()

    confidence = 100
    number_of_faces = 0

    # only show detections over 80% certainty
    for row in detections[0][0]:
        if row[2] > 0.80:
            confidence = round(row[2] * 100, 3)
            number_of_faces += 1
            x1, y1, x2, y2 = int(row[3] * img.shape[1]), int(row[4] * img.shape[0]), int(row[5] * img.shape[1]), int(
                row[6] * img.shape[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 6)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), number_of_faces, confidence


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

# Load shape predictors
hog_svm_shape_predictor = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')


# auxiliary methods
def apply_alpha_to_transparent_image(foreground: Image, alpha_of_masks: int) -> Image:
    foreground_copy = foreground.copy()
    foreground_copy.putalpha(alpha_of_masks)
    foreground.paste(foreground_copy, mask=foreground)
    return foreground


def find_face_rectangles_mtcnn(box_and_keypoints_list: list[tuple[list, dict]]) -> list[tuple[int, int, int, int]]:
    face_coordinates = []
    for (box, keypoints) in box_and_keypoints_list:
        box_upper_left_x = box[0]
        box_upper_left_y = box[1]
        box_width = box[2]
        box_height = box[3]
        face_coordinates.append(
            (box_upper_left_x, box_upper_left_y, box_width, box_height))
    return face_coordinates


def find_free_coordinates_outside_of_rectangles(img: Image, number_of_inserted_items: int, width_of_inserted_item: int,
                                                height_of_inserted_item: int, taken_coordinates=None,
                                                search_limit_per_try: int = 2500) -> list[tuple[int, int]]:
    if taken_coordinates is None:
        taken_coordinates = []

    width, height = img.size
    inserted_items = 0
    inserted_items_coords = []
    unsuccessful_tries = 0
    while inserted_items < number_of_inserted_items:
        if unsuccessful_tries >= search_limit_per_try:
            break

        item_x = random.randint(0, width - 1)
        item_y = random.randint(0, height - 1)
        inserted = True
        for coordinates in taken_coordinates:
            is_outside_of_rectangle = (item_x + width_of_inserted_item < coordinates[0]
                                       or item_x > coordinates[0] + coordinates[2]
                                       or item_y > coordinates[1] + coordinates[3]
                                       or item_y + height_of_inserted_item < coordinates[1])
            if not is_outside_of_rectangle:
                inserted = False
                break
        if not inserted:
            unsuccessful_tries += 1
            continue
        inserted_items += 1
        inserted_items_coords.append((item_x, item_y))
        taken_coordinates.append((item_x, item_y, width_of_inserted_item, height_of_inserted_item))
    return inserted_items_coords
