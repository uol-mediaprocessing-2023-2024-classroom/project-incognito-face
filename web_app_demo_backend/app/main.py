import os
import ssl
import urllib.request

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageFilter


app = FastAPI()

# SSL configuration for HTTPS requests
ssl._create_default_https_context = ssl._create_unverified_context

# CORS configuration: specify the origins that are allowed to make cross-site requests
origins = [
    "https://localhost:8080",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# A simple endpoint to verify that the API is online.
@app.get("/")
def home():
    return {"Test": "Online"}


@app.get("/get-blur/{cldId}/{imgId}")
async def get_blur(cldId: str, imgId: str, background_tasks: BackgroundTasks):
    """
    Endpoint to retrieve a blurred version of an image.
    The image is fetched from a constructed URL and then processed to apply a blur effect.
    """
    img_path = f"app/bib/{imgId}.jpg"
    image_url = f"https://cmp.photoprintit.com/api/photos/{imgId}.org?size=original&errorImage=false&cldId={cldId}&clientVersion=0.0.1-medienVerDemo"

    download_image(image_url, img_path)
    apply_blur(img_path)

    # Schedule the image file to be deleted after the response is sent
    background_tasks.add_task(remove_file, img_path)

    # Send the blurred image file as a response
    return FileResponse(img_path)


# Downloads an image from the specified URL and saves it to the given path.
def download_image(image_url: str, img_path: str):
    urllib.request.urlretrieve(image_url, img_path)


# Opens the image from the given path and applies a box blur effect.
def apply_blur(img_path: str):
    blurImage = Image.open(img_path)
    blurImage = blurImage.filter(ImageFilter.BoxBlur(10))
    blurImage.save(img_path)


# Deletes the file at the specified path.
def remove_file(path: str):
    os.unlink(path)


# Global exception handler that catches all exceptions not handled by specific exception handlers.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred."},
    )
