#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[32]:


# Refer https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
dataset=pd.read_csv("breast-cancer-wisconsin.data",header=None)
print(dataset.head())
#Convert missing value '?' as Nan. sklearn.impute.SimpleImputer will convert Nan to 'mean'
dataset.replace('?',np.nan,inplace=True)
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1:].values
print(X[:10])
print(y[:10])


# # Filling Missing data

# In[33]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X=imputer.transform(X)


# # Split dataset to training and test set

# In[34]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
print(X_train)
print(y_train)


# # Training Logistic Regression model in training set

# In[35]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# # Predicting test set result

# In[36]:


y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),(y_test.reshape(len(y_test),1))),1))


# # Making confusion matrix

# In[37]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[38]:


(109+60)/(109+3+3+60)


# In[ ]:




