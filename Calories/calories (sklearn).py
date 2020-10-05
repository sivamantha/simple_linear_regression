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

data = pd.read_csv('C:/Users/sivam/OneDrive/Projects/Simple Linear Regression/Calories/calories_consumed.csv')
data  
data.describe() 

""" create the regression """
y = data['Weight']
x1 = data['Calories']
x1.shape
y.shape
x = x1.values.reshape(-1,1)

""" regression itself """

reg = LinearRegression()
reg.fit(x,y)
reg.score(x,y) #R-squared
reg.coef_ # b1
reg.intercept_ # b0

#p-value
f_regression(x,y) 
p = f_regression(x,y)[1]
p.round(3)

""" predict data """
new_data = pd.DataFrame(data=[2258, 3587, 1245, 2555, 2222], columns = ['Calories'])
new_data
new_data['predicted_weight'] = reg.predict(new_data)
new_data


