#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Importing data

# In[18]:


data=pd.read_csv("50_Startups.csv")
print(data)
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1:].values
print(X)
print(Y)


# # Encoding Categorial data

# In[5]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder="passthrough")
#Fit and transform will encode "state" column in index 3 and place it in beginning(first)
X=np.array(ct.fit_transform(X))
print(X)


# # Splitting training and test data set

# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state =0)


# # Training Multiple Linear Regression model on training set

# In[10]:


from sklearn.linear_model import LinearRegression
# sklearn LinearRegression(Model selection) filter high Significant Value(SV) and eliminates least SV(high deviation value)
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
# Formula--> y=b0+b1x1+b2x2+..._bnxn
# We don't need to provide feature scaling in linear regression since we are adding coefficient
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


# # Predicting the test set

# In[17]:


y_pred=regressor.predict(X_test)
np.set_printoptions(precision=2) # Print only 2 decimal point
# y_pred.reshape(len(y_pred),1)--> len(y_pred)=No. of rows & 1 column. 
# concatenate=1st Arg - tuple of same length array & axis, 2nd Arg - axis(0-Horizontal & 1-Vertical)
# https://www.superdatascience.com/pages/ml-regression-bonus-2(https://colab.research.google.com/drive/1ABjLFzknByfU4-F4roa1hX36H3aZlu6J?usp=sharing)
print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))# Print value Vertically


# # Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')

# In[20]:


print(regressor.predict([[1.0,0.0,0.0,160000,130000,300000]]))
print(regressor.predict([[1.0,0.0,0.0,162597.7,151377.59,443898.53]]))


# # Getting the final linear regression equation with the values of the coefficients

# In[21]:


print(regressor.coef_)
print(regressor.intercept_)


# In[ ]:




