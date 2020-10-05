""" import the relavant libraries """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

""" load the data """

data = pd.read_csv('C:/Users/sivam/OneDrive/Projects/Simple Linear Regression/Delivery Time/delivery_time.csv')  
data.describe() 

""" create the regression """
y = data['Delivery Time']
x1 = data['Sorting Time']
x1.shape
y.shape
x = x1.values.reshape(-1,1)
plt.scatter(x1,y)

""" regression itself """

reg = LinearRegression()
reg.fit(x,y)
reg.score(x,y) #R-squared
reg.coef_ # b1
reg.intercept_ # b0

#p-value
f_regression(x,y) 
p = f_regression(x,y)[1]
p.round(6)

""" predict data """
new_data = pd.DataFrame(data=[10.9, 6.2, 0.5, 8.5, 30.7], columns = ['sorting_time'])
new_data
new_data['predicted_delivery_time'] = reg.predict(new_data)
new_data

