# import the relavant libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

#load the data
# using pandas will display data in tablaur format

data = pd.read_csv('C:/Users/sivam/OneDrive/Projects/Simple Linear Regression/Calories/calories_consumed.csv')
data  # used to display the data
data.describe() #output of mean, median, min, max, 1st & 3rd quadrants

#create the regression
y = data['Weight']
x1 = data['Calories']

#explore the data

plt.scatter(x1,y)
plt.xlabel('Calories', fontsize = 20)
plt.ylabel('Weight', fontsize = 20)
plt.show()

#regression itself

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

#plotting the regression line

plt.scatter(x1,y)
yhat = -625.7524 + 0.4202*x1
fig = plt.plot(x1,yhat, lw = 2, c='red', label = 'regression line')
plt.xlabel('Calories', fontsize = 20)
plt.ylabel('Weight', fontsize = 20)
plt.show()