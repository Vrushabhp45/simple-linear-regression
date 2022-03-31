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
#REGRESSION PLOT
sns.regplot(x=data_set['years_experience'],y=data_set['salary'])
plt.show()
#y= a+ b* log(x)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Input dataset
X_log = np.log(data_set['years_experience'].values.reshape(-1,1))

# Output or Predicted Value of data
y_log = data_set['salary'].values.reshape(-1,1)
X_train_log, X_test_1og, Y_train_log, Y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state= 42)
y_pred_log= LinearRegression()
y_pred_log.fit(X_train_log,Y_train_log)
LinearRegression()
print(" Intercept value of Model is " ,y_pred_log.intercept_)
print("Co-efficient Value of Log Model is : ", y_pred_log.coef_)
l_model= y_pred_log.predict(X_test_1og)
l_model

pmsh_pf_1 = pd.DataFrame({'Actual':Y_test_log.flatten(), 'Predict': l_model.flatten()})
pmsh_pf_1
plt.scatter(X_test_1og, Y_test_log,  color='gray')
plt.plot(X_test_1og, l_model, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_log, l_model))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test_log, l_model) )
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test_log, l_model)))
print("R^2 Score :          ", metrics.r2_score(Y_test_log, l_model))

