import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
#IMPORT DATA
data_set=pd.read_csv("C:/Users/HP/PycharmProjects/Excelrdatascience/Salary_Data.csv")
print(data_set.head())
#DATA VISUALIZATION
sns.distplot(data_set['Salary'])
plt.show()
sns.distplot(data_set['YearsExperience'])
plt.show()
# Renaming Columns
data_set=data_set.rename({'YearsExperience':'years_experience', 'Salary':'salary'},axis=1)
data_set
#correlation analysis
print(data_set.corr())
#REGRESSION PLOT
sns.regplot(x=data_set['years_experience'],y=data_set['salary'])
plt.show()
y=data_set["salary"].values
print(y.reshape((1,-1)))
x=data_set["years_experience"].values.reshape((-1,1))
print(x)
#CREATE MODEL
model = LinearRegression()
#FIT MODEL
model.fit(x, y)
model = LinearRegression().fit(x, y)
#GET THE RESULT
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept:', new_model.intercept_)
print('slope:', new_model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept:', new_model.intercept_)
print('slope:', new_model.coef_)
#PREDICT RESPONSE
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')