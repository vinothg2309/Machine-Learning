#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[43]:


dataset=pd.read_csv("Position_Salaries.csv")
print(dataset)
X=dataset.iloc[:,1:-1]
Y=dataset.iloc[:,-1]
print(X)
print(Y)


# # Training Linear Regression model on whole dataset

# In[44]:


from sklearn.linear_model import LinearRegression
#It involves only 1 dependent & independent variable. Formula--> y=mx+b
linear_reg = LinearRegression()
linear_reg.fit(X,Y)


# # Training Polynominal Regression model on whole dataset

# In[45]:


#Polynomial Regression Formula-->y=b0+b1x1+b2x1^2+b3x1^3+...bnx1^n(x1^-x1 power)
# We don't need to provide feature scaling in linear regression since we are adding coefficient
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4) #degree(d)=1 represents y=b0+b1x1+b2x1^2. d=3 then upto b3x1^3
x_poly=poly_reg.fit_transform(X) # Adding X value => each level(column) is powered by degree(4)mentioned in above line
print(x_poly)
linear_reg_2=LinearRegression()
linear_reg_2.fit(x_poly,Y)


# # Visualising Linear Regression results

# In[46]:


plt.scatter(X,Y,color="red")
plt.plot(X, linear_reg.predict(X), color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()


# # Visualising Polynominal Regression results

# In[47]:


plt.scatter(X,Y,color="red")
plt.plot(X, linear_reg_2.predict(x_poly), color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()


# # Visualising Polynominal Regression results(for higher resolution and smoother curve) 

# In[50]:


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# # Predicting new result with Linear Regression

# In[52]:


linear_reg.predict([[6.5]])


# # Predicting new result with Polynomial Regression

# In[55]:


#High overfitting since we are considering level 10 which has huge deviation than other points
linear_reg_2.predict(poly_reg.fit_transform([[6.5]]))


# In[ ]:




