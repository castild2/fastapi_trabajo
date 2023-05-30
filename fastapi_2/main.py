import tensorflow as tf
from fastapi import FastAPI, Form
from image_process import process_image
from output_process import process_output
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
 "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model('model_resnet.h5')


# run with uvicorn main:app --reload

@app.post('/predict/')
async def predict(image: str = Form(...)):
    image = process_image(image)
    prediction = MODEL.predict(image)
    result = process_output(prediction)
    return result



