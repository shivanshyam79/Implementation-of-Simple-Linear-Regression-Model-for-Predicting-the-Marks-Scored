# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: shyam R
RegisterNumber:212223040200 
*/
```
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

## Ou![1](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/e2f95ce2-3eb1-4bff-a82e-14ee1e4306dc)
tput:![2](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/d4f05be1-18b8-4893-b72e-8da1c725a400)![3](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/75329092-0432-4066-8270-44033e68f158)![229978994-b0d2c87c-bef9-4efe-bba2-0bc57d292d20](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/2c73d9e1-3725-48da-8ef1-65704ec67590)![229979053-f32194cb-7ed4-4326-8a39-fe8186079b63](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/ae2d7421-fd6e-4760-914a-215acb9bf960)![229979114-3667c4b7-7610-4175-9532-5538b83957ac](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/7612ce1e-4aff-49c3-a7b6-2ef4f4c6464c)
![229979169-ad4db5b6-e238-4d80-ae5b-405638820d35](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/a74eeba6-2cc6-447a-a54f-d09f88f1bf25)![229979225-ba9085![229979276-bb9ffc68-25f8-42fe-9f2a-d7187753aa1c](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/a35e4fc4-7b4b-42a3-84f6-59c587cbe5ce)
3c-7fe0-4fb2-8454-a6a0b921bdc1](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/15c788fb-bb34-472d-b60e-60ac2b5b56f1)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
