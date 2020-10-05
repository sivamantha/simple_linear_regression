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

data = pd.read_csv('C:/Users/sivam/OneDrive/Projects/Simple Linear Regression/Employee Data/emp_data.csv')  
data.describe() 
data
""" create the regression """
y = data['Churn_out_rate']
x1 = data['Salary_hike']
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
p.round(4)

""" predict data """
new_data = pd.DataFrame(data=[200,900,2000,3000,1500], columns = ['salary_hike'])
new_data
new_data['churn_out_rate'] = reg.predict(new_data)
new_data
 