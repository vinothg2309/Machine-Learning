#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Importing Data

# In[18]:


dataset=pd.read_csv("Position_Salaries.csv")
print(dataset)
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1:].values #[:,-1:]-->Returns 2D array, [:,-1] --> Returns 1D array
print(X)
print(y)

#Feature Scaling(StandardScaler) expect 2D as input. so converting 1D to 2D as below
#y=y.reshape(len(y),1)#reshape(no.of row, no. of column)
#print(y)


# # Feature Scaling

# In[12]:


# Feature scaling is applied in SVR since v don't have coefficient as in Linear (or) polynomial regression
# Feature Scaling - x represents independent data set
# Standardisation--> x-mean(x)/standard deviation(x) --> Range (-3 to +3)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
sc_y=StandardScaler()
y=sc_y.fit_transform(y)
print(X)
print(y)


# # Training SVR on whole dataset

# In[13]:


from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)


# # Predicting a new result

# In[14]:


# inverse_transform-Convert scaled value back to actual value. sc_X.transform-tranform actual to scaled value
# We are ignoring level 10(CEO) which has huge deviation than other points.
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))


# In[15]:


sc_y.inverse_transform(regressor.predict(sc_X.transform([[10]])))


# # Visualising SVR results

# In[16]:


# inverse_transform should be applied to X and y to Convert scaled value back to actual value
# SVR ignores level 10 since it doesn't fit boundary(decision boundary)
# Regression line is called Hyperplane and its surrounded by decision boundary - decision boundary
# https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color="red")
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color="blue")
plt.title("Truth or Bluff (Support Vector Regression)")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()


# In[17]:


X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.01) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color="red")#this step required because data is feature scaled.
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color="blue")
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:




