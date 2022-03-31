import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
#PROVIDE DATA
data_set=pd.read_csv("C:/Users/HP/PycharmProjects/Excelrdatascience/delivery_time.csv")
print(data_set.head())
#DATA VISUALIZATION
sns.distplot(data_set['Delivery Time'])
plt.show()
sns.distplot(data_set['Sorting Time'])
plt.show()
# EDA
data_set=data_set.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
data_set
#CORRELATION ANALISYS
print(data_set.corr())
#REGRESSION PLOTTING
sns.regplot(x=data_set['sorting_time'],y=data_set['delivery_time'])
plt.show()
#DEFINING DATASET (DEPENDENT AND INDEPENDENT)
y=data_set["delivery_time"].values
print(y.reshape((1,-1)))
x=data_set["sorting_time"].values.reshape((-1,1))
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