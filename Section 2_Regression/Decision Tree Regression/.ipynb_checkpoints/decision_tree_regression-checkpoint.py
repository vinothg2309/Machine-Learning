#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dataset=pd.read_csv("Position_Salaries.csv")
print(dataset)
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values #[:,-1:]-->Returns 2D array, [:,-1] --> Returns 1D array


# # Training Decision Tree Regression on the whole dataset

# In[6]:


# Decision Tree Regression(DTR) doesn't requires Feature scaling since it splits the data into multiple leafs and work on it
# DTR isn't good fit for 2D Array(Level & Salary)
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


# # Predicting a new result

# In[7]:


regressor.predict([[6.5]])


# # Visualising Decision Tree Regression Prediction(High Resolution)

# In[11]:


X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:




