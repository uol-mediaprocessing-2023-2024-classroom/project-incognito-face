import os
import ssl
import dlib
import urllib.request

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from PIL import Image, ImageFilter
from pathlib import Path
from io import BytesIO
import cv2
import numpy

app = FastAPI()
IMAGE_PATH = Path(__file__).parent.parent.parent / 'images'

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


@app.get('/get-img/{img_name}')
async def get_img(img_name: str):
    return FileResponse(IMAGE_PATH / img_name)


@app.get('/get-img-data/{img_name}')
async def get_img_data(img_name: str):
    return JSONResponse(content=get_image_data(IMAGE_PATH / img_name))


@app.get('/get-blur/{img_name}')
async def get_blur(img_name: str):
    img = Image.open(IMAGE_PATH / img_name)
    img = apply_blur(img)
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    return StreamingResponse(BytesIO(img_bytes_io.getvalue()))


@app.get('/get-face-data/{img_name}')
async def get_face_data(img_name: str):
    # Percentages are just dummies rn
    response = {
        'viola-jones': {
            'url': f'get-face/viola-jones/{img_name}',
            'percentage': '100'
        },
        'hog-svn': {
            'url': f'get-face/hog-svn/{img_name}',
            'percentage': '100'
        }
    }
    return JSONResponse(content=response)


@app.get('/get-face/viola-jones/{img_name}')
async def get_face_viola_jones(img_name: str):
    img = Image.open(IMAGE_PATH / img_name)
    img = highlight_face_viola_jones(img)
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    return StreamingResponse(BytesIO(img_bytes_io.getvalue()))


@app.get('/get-face/hog-svn/{img_name}')
async def get_face_hog_svn(img_name: str):
    img = Image.open(IMAGE_PATH / img_name)
    img = highlight_face_hog_svm(img)
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    return StreamingResponse(BytesIO(img_bytes_io.getvalue()))


def get_image_data(img_path):
    return {
        'name': img_path.name,
        'timestamp': os.path.getmtime(img_path),
        'url': f'get-img/{img_path.name}',
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
    return Image.fromarray(img_rgb)


def highlight_face_hog_svm(img: Image):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray_image)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 6)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# Global exception handler that catches all exceptions not handled by specific exception handlers.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={'message': 'An unexpected error occurred.'},
    )
