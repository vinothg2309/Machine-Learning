#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Data.csv')


# In[13]:


# Goal is to retrieve whether purchase will happen based on Country,Age,Salary
# X represent - Metrics(Country,Age,Salary)
X = dataset.iloc[:,:-1].values

# Y represent - Dependent Variable which v need to retreive(Purchased)
Y = dataset.iloc[:,-1:].values

print(X)
print(Y)


# # Filling missing data

# In[14]:


# Filling missing data by average using scikit
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3]) # check missing value & computes mean value of the columns
X[:,1:3] = imputer.transform(X[:,1:3]) # uses a previously computed mean to transform/autoscale the data
print(X)


# # Encoding Categorial data

# In[15]:


# Encoding Categorial data - Independent Variable(Country,Age,Salary)
# Encoding independent variable(Country) since it isn't numeric
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# We need to transform only Country remaining column should be impacted. so passing  remainder='passthrough'
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#Encoding dependent variable
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y = le.fit_transform(Y)
print(Y)


# # Splitting training and test data set

# In[16]:


# Splitting dataset to training set&test set.
from sklearn.model_selection import train_test_split
# 20% to test set in random row and 80% to training set.
# X_test & Y_test will map to same row. Same for training as well
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state =1)
print('X_train\n', X_train)
print('X_test\n', X_test)
print('Y_train\n', Y_train)
print('Y_test\n', Y_test)


# # Feature Scaling

# Using the transform method we can use the same mean and variance as it is calculated from our training data to transform our test data. 
# Thus, the parameters learned by our model using the training data will help us to transform our test data. 
# If we will use the fit method on our test data too, we will compute a new mean and variance that is a new scale for each feature and will let our model learn about our test data too. 
# https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe

# In[17]:


# Feature Scaling - x represents independent data set
# Standardisation--> x-mean(x)/standard deviation(x) --> Range (-3 to +3)
# Normalisation--> x-min(x)/max(x)-min(x) --> Range (0 to 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# we shouldn't standardize Country which is done in OneHotEncoder since it return range(-3 to +3), we will loose country
# Only apply standardisation to only numeric value
# fit-calculate means&SD, transform-transforming all the features using the respective mean and variance
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
# X_test is future incoming data for which we need to predict dependent variable(Purchased column).
# we will only transform it. We need to use same scalar object(sc) as that of train 
#Using the transform method we can use the same mean and variance as it is calculated from our training data to transform our test data. 
#Thus, the parameters learned by our model using the training data will help us to transform our test data.
X_test[:,3:]=sc.transform(X_test[:,3:])
print('X_train\n', X_train)
print('X_test\n', X_test)


# In[ ]:





# In[ ]:




