import os
import sys

# Flask
from flask import Flask, request, render_template, Response, jsonify
#from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil
from gradcam import create_gradcam_model, compute_gradcam


# Declare a flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


MODEL_PATH = 'models/oldModel.h5'
model = load_model(MODEL_PATH)
grad_model = create_gradcam_model(model)

print('Model and Grad-CAM loaded. Start serving...')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        payload = request.json
        if isinstance(payload, dict):
            img_base64 = payload.get('image', '')
        else:
            img_base64 = payload

        orig_pil = base64_to_pil(img_base64)
        
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        img_path = os.path.join(uploads_dir, 'image.jpg')
        orig_pil.save(img_path)

        # -------------------------------------------------------------------------
        # VIVA EXPLANATION: Prediction & Grad-CAM Pipeline
        # 1. Image is converted from Base64 string to RGB PIL Image.
        # 2. compute_gradcam() computes both the binary inference score and the
        #    gradient-weighted class activation map (Grad-CAM) from the final conv layer.
        # 3. Probabilities and risk stratification are returned alongside visual overlays.
        # -------------------------------------------------------------------------
        res = compute_gradcam(orig_pil, model, grad_model=grad_model)
        print(f"Prediction: {res['prediction']} ({res['confidence']}%) - Score: {res['score']} - Band: {res['confidence_band']}")

        return jsonify(
            result=res['prediction'],
            confidence=res['confidence'],
            score=res['score'],
            pneumonia_prob=res['pneumonia_prob'],
            normal_prob=res['normal_prob'],
            confidence_band=res['confidence_band'],
            confidence_band_class=res['confidence_band_class'],
            driving_class=res['driving_class'],
            driving_caption=res['driving_caption'],
            overlay_image=res['overlay_base64'],
            heatmap_image=res['heatmap_base64'],
            original_image=res['original_base64'],
            attention_summary=res['attention_summary']
        )

    return jsonify(error="Invalid request"), 400




if __name__ == '__main__':
    print('Starting Flask server at http://127.0.0.1:5002')
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=False)


