# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: shyam R
RegisterNumber:212223040200 */

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
## Dataset
![dataset](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/8f6358d6-74a0-40b4-a88b-ac723b71f1f2)
## Head Values
![head](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/941f8428-1463-4d50-842c-02dddf1f8c95)
## Tail Values
![tail](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/1b4f95d4-459c-4b16-a0c6-3901e9fbfbcb)
## X and Y values
![xyvalues](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/85242e23-27f9-4977-99e7-4891c5b6f67b)
## predication values of X and Y
![predict ](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/107747e0-e7b5-46c7-b7ec-064e3c5c8af0)
## MSE,MAE and RMSE
![values](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/781388a5-ab68-4cc3-b2e4-53d750dbd780)
## Training Set
![train](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/51e8ce78-2f50-46de-bb8b-02e5397c80f5)
## Testing Set
![test](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/07f145ad-5c13-4823-9ba3-55770cd656d9)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
