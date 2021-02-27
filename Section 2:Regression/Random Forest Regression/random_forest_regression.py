#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv("Position_Salaries.csv")
print(dataset)
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values #[:,-1:]-->Returns 2D array, [:,-1] --> Returns 1D array


# # Training Random Forest Regression on the whole dataset

# In[3]:


# Break dataset to set of decision tree and perform regression in it
# RandomForestRegressor isn't good fit for 2D Array(Level & Salary). works fine in data with multiple dimension
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10, random_state=0) # 10 decision tree will b created in dataset
regressor.fit(X,y)


# # Predicting a new result

# In[4]:


regressor.predict([[6.5]])


# # Visualising Random Forest Regression Prediction(High Resolution)

# In[5]:


X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:




