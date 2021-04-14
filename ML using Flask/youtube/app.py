# https://github.com/krishnaik06/DL-Project-For-Beginner

import numpy as np

#Keras
from keras.applications.vgg19 import preprocess_input, decode_predictions
#import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
# coding=utf-8
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob
import re
from werkzeug.utils import secure_filename


app = Flask(__name__)
model_path = 'vgg19.h5'

# Load model
model = load_model(model_path)
#model.make_predict_function() # Necessary only fr VGG

def model_predict(image_path, model):
    #VGG, Resnet Transfer learning expects 224,224 image size & works fine on the same
    img = image.load_img(image_path, target_size=(224,224)) 
    #Preprocessing the image
    img1 = image.img_to_array(img)
    print(img.shape)
    img = img1.expand_dims(img1,axis=0)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    img = preprocess_input(img)
    pred = model.predict(img)
    return pred
    
@app.route('/', methods=['GET'])
def index():
    render_template('index.html')
    
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
         # Get the file from post request
         f = request.files['file'] # file is name of uploaded file in UI
         # Save the file to ./uploads
         basepath = os.path.dirname(__file__)  # Returns current working directory
         file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
         f.save(file_path)
         
         #Make Prediction
         preds = model_predict(file_path, model)
         
         
         # Process your result for human
         # pred_class = preds.argmax(axis=-1)  
         pred_class = preds.argmax(axis=-1)
         result = str(pred_class[0][0][1])
         return result
    return None 

if __name__ == '__main__':
    app.run(debug=True)