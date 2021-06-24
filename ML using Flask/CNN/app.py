# -*- coding: utf-8 -*-

import os, glob, sys, re
import numpy as np

#Keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('model_vgg19.h5')

def model_predict(filepath):
    # target_size is mentioned as (224,224) while training the mode in CNN And Transfer Learning.ipynb
    img = image.load_img(filepath, target_size=(224,224))
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x,axis=0)
    print(x.shape)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds="The Person is Infected With Pneumonia"
    else:
        preds="The Person is not Infected With Pneumonia"
    
    
    return preds

@app.route('/', methods=['GET'])
def index():
    render_template('index.html')
    

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        # Get file from request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make Prediction
        preds = model_predict(file_path)
        return preds



if __name__=='main':
    app.run()