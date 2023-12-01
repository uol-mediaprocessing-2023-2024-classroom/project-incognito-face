import os
import ssl
import dlib
import base64
import urllib.request
import concurrent.futures 

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from PIL import Image, ImageFilter
from pydantic import BaseModel
from pathlib import Path
from io import BytesIO
import cv2
import numpy

app = FastAPI()
IMAGE_PATH = Path(__file__).parent.parent.parent / 'images'
FACE_DETECTION_ALGORITHMS = [
        {
            'name': 'viola-jones',
            'displayName': 'Viola Jones'
        },
        {
            'name': 'hog-svn',
            'displayName': 'HOG-SVN'
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
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    return JSONResponse(content={'base64': f'data:image/png;base64,{base64.b64encode(img_bytes_io.getvalue()).decode("utf-8")}'})

class RunFaceDetectionRequestData(BaseModel):
    base64: str

@app.post('/run-face-detection')
async def get_face_data(data: RunFaceDetectionRequestData):
    result = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_algorithm, algorithm, data.base64) for algorithm in FACE_DETECTION_ALGORITHMS]
        for future in concurrent.futures.as_completed(futures):
            try:
                result.append(future.result())
            except Exception as e:
                print(f'An error occurred while trying to run multi-threaded face detection: {e}')
    return JSONResponse(content=result)

def process_algorithm(algorithm, img):
    img = Image.open(BytesIO(base64.b64decode(img[22:])))
    match algorithm['name']:
        case 'viola-jones':
            img, has_face, confidence = highlight_face_viola_jones(img)
        case 'hog-svn':
            img, has_face, confidence = highlight_face_hog_svm(img)
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    return {
        'name': algorithm['name'],
        'base64': f'data:image/png;base64,{base64.b64encode(img_bytes_io.getvalue()).decode("utf-8")}',
        'has_face': has_face,
        'confidence': confidence
    }


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


def highlight_face_viola_jones(img: Image):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 6)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), True, 100


def highlight_face_hog_svm(img: Image):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray_image)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 6)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), False, 50


# Global exception handler that catches all exceptions not handled by specific exception handlers.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={'message': 'An unexpected error occurred.'},
    )
