from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
import tf_keras as k3
import keras
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from utils.feature_extractor import preprocess, decode_prediction, num_to_label

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = r"D:\flask app\model\new-model.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = k3.models.load_model(model_path, compile=False)
print('Model loaded successfully.')



@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict
            image = preprocess(filepath)
            print(image.shape)
            image = image/255.
            preds = model.predict(image.reshape(1, 256, 64, 1))
            prediction = decode_prediction(preds)
            prediction = num_to_label(prediction)
            print(prediction)

            image_path = f'/static/uploads/{filename}'

    return render_template('index.html', prediction=prediction, image_path=image_path)
    


app.run(host='0.0.0.0', port=5000, debug=True)