#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################### https://github.com/krishnaik06/FastAPI ##################
####################### Swagger URL - http://localhost:8000/docs #########
####################### http://localhost:8000/redoc
### /home/vinoth/Learning/Python/Workouts/ML-Udemy-Git/Docker/Banker Note Authentication.ipynb
"""
Created on Fri Jun 25 08:53:39 2021

@author: vinoth
"""
# 1. Import libraries
import uvicorn #ASGI
from fastapi import FastAPI
from BankNote import BankNote
import numpy as np
import pickle
import pandas as pd

# 2. Create app object
app = FastAPI()
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message':'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/test/{name}')
def get_name(name:str):
     return {'Welcome To App': f'{name}'}
 
# It requires mandatory request param 'name' which needs to be passed from UI
# V can use openAPI(swagger-ui) which comes out of box via http://localhost:8000/docs
@app.get('/Welcome')
def welcome(name:str):
    return {'Welcome To App':name}
 
# 5. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_bankenote(data:BankNote):
    print('Data from UI : ', data)
    data = data.dict()
    print('Data dict: ', data)
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    print('variance: {} \nskewness: {}\ncurtosis: {}\nentropy: {}'.format(variance,skewness,curtosis,entropy))
    predict = classifier.predict([[variance,skewness,curtosis,entropy]])
    if predict[0] > 0.5:
        prediction = "Fake Note"
    else:
        prediction = "It's a Bank Note"
    
    return {'Prediction': prediction}

# 6. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port='8000')
    
    
###### cmd - uvicorn main:app --reload
# uvicorn app:app --reload