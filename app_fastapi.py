# use FastAPI to deploy model
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import nest_asyncio
import uvicorn
import os

app = FastAPI()

@app.get("/")
def home():
    return "Succuessful!"

model = load_model('model.h5')

def predict_image(img: Image.Image):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    prediction = predict_image(img)
    confidence = prediction[0]
    predicted_index = np.argmax(confidence)
    
    class_names = ['black', 'grizzly', 'panda', 'polar', 'teddy']
    predicted_class = class_names[predicted_index]
    confidence = confidence[predicted_index]
    
    return {"predicted_class": predicted_class, 
                    "confidence": round(float(confidence), 3)}

nest_asyncio.apply()
host = "127.0.0.1"
uvicorn.run(app, host=host, port=8000)