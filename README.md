# bear-classify
Classify 5 classes of bears: training, inference and deploy
# Dataset
https://www.kaggle.com/datasets/hoturam/bear-dataset
# Program list
* `app_fastapi.py`: use FastAPI to deploy the model  
* `app_flask.py`: use Flask to deploy the model  
* `app_gradio.py`: use Gradio to make an UI  
* `predict.py`: inference  
* `train.py`: train model  
# Training model
* Run `train.py`
# Inference
* Run `predict.py`
# Use API to inference  
* Run `app_fastapi.py` or `app_flask.py`
* Go to command prompt and enter the following command:
  ```
  curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@your_image_path"
  ```
* If you use `app_fastapi.py`, you also can go to `http://127.0.0.1:8000/docs` to upload your image to inference (test API)
# Use User Interface to inference
* Run `app_gradio.py`
* Example:
  ![](https://github.com/eunice1226/bear-classify/blob/main/gradio_demo.png)
