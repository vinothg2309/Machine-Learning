# -*- coding: utf-8 -*-
# <--------------https://github.com/krishnaik06/Dockers---------------->

from flask import Flask, request
import pandas as pd
import pickle
from flasgger import Swagger

app = Flask(__name__)
# Access swagger in http://127.0.0.1:5000/apidocs
Swagger(app)

classifier = pickle.load(open('classifier.pkl', 'rb'))

@app.route('/')
def welcome():
    return "Welcome all"
    
# Testing URL
# http://127.0.0.1:5000/predict?variance=3&skewness=8&curtosis=-2&entropy=-1 ---> 0
# http://127.0.0.1:5000/predict?variance=-5&skewness=9&curtosis=-0.3&entropy=-5 ---> 1
@app.route('/predict')
def predict_note_authentication():
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])  
    return "The predicted value is "+str(prediction)

# Testing via postman --> URL: http://127.0.0.1:5000/predict_file, Body -> form-data -> set 'key' to file
@app.route('/predict_file', methods=['POST'])
def predict_note_file():
    
    """Let's authenticate the Banker note
    This is using docstring for specifications.
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
            
    """
    df_test = pd.read_csv(request.files.get("file"))
    prediction_values = classifier.predict(df_test)
    return "The predicted value for csv is "+str(list(prediction_values))
    

    
if __name__ == '__main__':
    app.run(host='0.0.0.0')