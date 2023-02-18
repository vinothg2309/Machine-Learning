#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:57:42 2021

@author: vinoth
"""


import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('hiring.csv')
df.head()

empty = df.isnull().sum()
df['experience'].fillna(0, inplace=True)
df['test_score'].fillna(df['test_score'].mean(), inplace=True)

X = df.iloc[:,:-1]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

print(type(X))

X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

y = df.iloc[:,-1:]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))