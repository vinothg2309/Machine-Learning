#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Importing Data

# In[20]:


data = pd.read_csv("Salary_Data.csv")
print(data)
#Indepent Variable
X=data.iloc[:,0:-1].values
Y=data.iloc[:,-1:].values
print(X)
print(Y)


# # Spliting dataset to training and test set

# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
print(X_train)


# # Training Simple Linear Regression model on training set

# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train) # Passing train data


# # Predicting the test set Result

# In[12]:


y_pred=regressor.predict(X_test) # We need to predict salary(Y) based on No.of years(X)


# # Visualising Training set result

# In[15]:


plt.scatter(X_train, Y_train, color='red') # Plotting salary & No. of years
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience(Training Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()


# # Visualising test set result

# In[16]:


plt.scatter(X_test, Y_test, color='red') # Plotting salary & No. of years
# Predictedsalary of training set should be same as test set in Regression(blue) line. We wont replace train with test
plt.plot(X_train, regressor.predict(X_train), color='blue') 
plt.title("Salary vs Experience(Test Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()


# # Making a single prediction (for example the salary of an employee with 12 years of experience)

# In[21]:


print(regressor.predict([[12]]))
print(regressor.predict([[10]]))


# # Getting the final linear regression equation with the values of the coefficients

# In[19]:


# y = mx+b --> m=coefficient, b=interceptor, x=Exp. to which salary is to be determined
# We don't need to provide feature scaling in linear regression since we add coefficient in linear regression
print(regressor.coef_)
print(regressor.intercept_)
# Salary=9345.94×YearsExperience+26816.19


# In[ ]:




