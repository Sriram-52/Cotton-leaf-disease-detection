
from __future__ import division, print_function
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


app = Flask(__name__)


MODEL_PATH = 'models/model_vgg19.h5'


model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)

    x = x/255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        return "The leaf is diseased cotton leaf"
    elif preds == 1:
        return "The leaf is diseased cotton plant"
    elif preds == 2:
        return "The leaf is fresh cotton leaf"
    else:
        return "The leaf is fresh cotton plant"


@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        file_path = 'uploads/' + f.filename
        f.save(file_path)

        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5000, debug=True)
