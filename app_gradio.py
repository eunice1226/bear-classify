# use Gradio to build a user interface
import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    
    class_names = ['black', 'grizzly', 'panda', 'polar', 'teddy']
    confidences = predictions[0]
 
    predicted_index = np.argmax(confidences)
    predicted_class = class_names[predicted_index]
    confidence = confidences[predicted_index]
    
    return predicted_class, round(confidence, 3)


iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Class"),
        gr.Number(label="Accuracy")
    ],
    title="熊熊種類辨識系統",
    description="請上傳熊熊的影像，模型將進行辨識並顯示結果。"
)

iface.launch()