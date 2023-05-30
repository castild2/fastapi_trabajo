import base64
import io
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_WIDTH = 128
IMAGE_HEIGTH = 128


def process_image(b64_image):
    image_data = base64.b64decode(b64_image)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGTH))
    image = np.array(image).reshape((1, IMAGE_WIDTH, IMAGE_HEIGTH, 3))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image
