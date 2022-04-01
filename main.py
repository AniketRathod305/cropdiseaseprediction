from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

crop_disease_model=tf.keras.models.load_model("crop_disease_final.h5")

class_names=["Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight","Tomato_healthy"]


@app.get("/hello")
async def hello():
    return "hey server started hii"


def read_file_as_img(data):
    image=np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file : UploadFile = File(...)):
    
    img=read_file_as_img(await file.read())
    
    img_batch=np.expand_dims(img,0)

    result=crop_disease_model.predict(img_batch)
    
    finalprediction=class_names[np.argmax(result[0])]
    confidence=np.max(result[0])

    return {
        'class':finalprediction,
        'confidence':float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
