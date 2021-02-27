import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Data.csv')

# Goal is to retrieve whether purchase will happen based on Country,Age,Salary
# X represent - Metrics(Country,Age,Salary)
X = dataset.iloc[:,:-1].values

# Y represent - Dependent Variable which v need to retreive(Purchased)
Y = dataset.iloc[:,-1:].values


# Filling missing data by average using {Pandas}
#average_salary = int(dataset['Salary'].mean())
#dataset['Salary'].fillna(average_salary, inplace = True)

#average_age = int(dataset['Age'].mean())
#dataset['Age'].fillna(average_salary, inplace = True)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3]) # check missing value & computes mean value of the columns
X[:,1:3] = imputer.transform(X[:,1:3]) # uses a previously computed mean to transform/autoscale the data

# Encoding Categorial data - Independent Variable(Country,Age,Salary)
# Encoding independent variable(Country) since it isn't numeric
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# We need to transform only Country remaining column should be impacted. so passing  remainder='passthrough'
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Encoding dependent variable
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y = le.fit_transform(Y)

# Splitting dataset to training set&test set.
from sklearn.model_selection import train_test_split
# 20% to test set in random row and 80% to training set.
# X_test & Y_test will map to same row. Same for training as well
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state =1)


# Feature Scaling - x represents independent data set
# Standardisation--> x-mean(x)/standard deviation(x) --> Range (-3 to +3)
# Normalisation--> x-min(x)/max(x)-min(x) --> Range (0 to 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# we shouldn't standardize Country which is done in OneHotEncoder since it return range(-3 to +3), we will loose country
# Only apply standardisation to only numeric value
# fit-calculate mean&SD, transform will convert to Standardisation
X_train[:,3:]=sc.fit_transform(X_train[:,3:])






