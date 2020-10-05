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

data = pd.read_csv('C:/Users/sivam/OneDrive/Projects/Simple Linear Regression/Salary/Salary_Data.csv')  
data.describe() 
data
""" create the regression """
y = data['Salary']
x1 = data['YearsExperience']
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
p.round(10)

""" predict data """
new_data = pd.DataFrame(data=[2, 9, 11, 14, 5], columns = ['Experience'])
new_data
new_data['salary'] = reg.predict(new_data)
new_data
