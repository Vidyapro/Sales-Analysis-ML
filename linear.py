# Loading the Data and Importing Libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Loading dataset

advert = pd.read_csv('Advertising.csv')
advert.head()

advert.info()

#Remove the index column

advert.columns

advert.drop(['Unnamed: 0'], axis=1, inplace = True)
advert.head()

# Exploratory Data Analysis

import seaborn as sns
sns.distplot(advert.sales)

sns.distplot(advert.newspaper)


sns.distplot(advert.radio)


sns.distplot(advert.TV)

# Exploring Relationships between Predictors and Response

sns.pairplot(advert, x_vars=['TV', 'radio', 'newspaper'], y_vars= 'sales', height=7,
            aspect=0.7, kind='reg')

advert.TV.corr(advert.sales)

advert.corr()

sns.heatmap(advert.corr(), annot=True)


# Creating the Simple Linear Regression Model


x = advert[['TV']]
x.head()

print(type(x))
print(x.shape)

y = advert.sales
print(type(y))
print(y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(x_train,y_train)

# Interpreting Model Coefficients

print(linreg.intercept_)
print(linreg.coef_)

# Making Predictions with our Model

y_pred = linreg.predict(x_test)
y_pred[:5]

# Model Evaluation Metrics

true = [100,50,30,20]
pred = [90,50,50,30]


# **Mean Absolute Error** (MAE) i

print((10 + 0 + 20 + 10)/4)

from sklearn import metrics
print(metrics.mean_absolute_error(true,pred))


# **Mean Squared Error** (MSE) 

print((10**2 + 0**2 + 20**2 + 10**2)/4)

print(metrics.mean_squared_error(true,pred))


# **Root Mean Squared Error** (RMSE)

print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4))

print(np.sqrt(metrics.mean_squared_error(true,pred)))

print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

