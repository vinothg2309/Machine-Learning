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
import uvicorn #ASGI - Asynchronous Server Gateway Interface
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI
from customer import Customer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import SpaceImputeTransformer

# 2. Create app object
app = FastAPI()
pickle_in = open('pipeline_model.pkl','rb')
classifier = joblib.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello World !!!'}

#{ "customerID": "5575-GNVDE", "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 34, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "Yes", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No", "PaymentMethod": "Mailed check", "MonthlyCharges": 56.95, "TotalCharges": "1889.5"}
@app.post('/predict')
def predict(customer: Customer):
    print('Data from UI : ', customer)
    data = customer.dict()
    print('Data : ', data)
    df1 = pd.DataFrame.from_dict([data], orient='columns')
    print('Pipeline : ', classifier)
    print('df1 : ', df1)
    pred = classifier.predict(df1)
    return {'Prediction': pred}

# 6. Run the API with uvicorn
#    Will run on http://127.0.0.1:8001
if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1", port='8001')
    
    
##### cmd - uvicorn main:app --reload
# uvicorn app:app --reload