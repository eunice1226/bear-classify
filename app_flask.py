# use Flask to deploy model
from flask import Flask, request, jsonify
import io
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model('model.h5')

def predict_image(img: Image.Image):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

@app.route("/", methods=["GET"])
def home():
    return "Successful!"

@app.route("/predict/", methods=["POST"])
def predict():
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    prediction = predict_image(img)
    confidence = prediction[0]
    predicted_index = np.argmax(confidence)
    
    class_names = ['black', 'grizzly', 'panda', 'polar', 'teddy']
    predicted_class = class_names[predicted_index]
    confidence = confidence[predicted_index]
    
    return jsonify({"predicted_class": predicted_class, 
                    "confidence": round(float(confidence), 3)})

if __name__ == "__main__":
    app.run(host)
