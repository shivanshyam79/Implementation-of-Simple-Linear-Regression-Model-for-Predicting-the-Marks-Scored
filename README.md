# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries.
#### 2.Set variables for assigning dataset values.
### 3.Import linear regression from sklearn.
### 4.Assign the points for representing in the graph.
### 5.Predict the regression for marks by using the representation of the graph.
### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

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
## df.head()
![263019075-db934862-eeeb-462b-aeb7-28b869113226](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/c2636c6f-9b15-4c61-b9b6-a7a2be08690e)
## df.tail()
![263019181-42e2a9f0-638c-40c5-9ab7-1ebc5b593d69](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/f6ebe1da-17ff-4cb0-ab76-d5e391e90b62)
## ARRAY VALUE OF X
![263019904-86c89c69-0df7-499e-9805-88444385fd12](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/122559c6-e8ae-4c3e-93c8-df1437572be3)
## ARRAY VALUE OF Y
![263022408-9afae483-751b-4a77-be53-0c2cc3e73204](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/655304e9-1ca7-4d1c-98a2-2d32cc5f212e)
## VALUES OF Y PREDICTION
![263025081-34ad6afe-5ee1-47bc-a293-e104dc4c0ccb](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/ae22943e-b183-4935-b9f6-d78f9e77e830)
## ARRAY VALUES OF Y TEST
![263025992-f7be3dfb-b4b5-44d7-9ca3-06812f202b9f](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/1d4f3719-fd73-4bb1-9da3-53d7473f08e9)
## TRAINING SET GRAPH
![263026832-82141809-4794-44e3-a164-aa8991f22e23](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/2195d499-3285-4b3c-8f1f-4cf1e8d99601)
## TEST SET GRAPH
![263027531-15efd5e6-ebfd-4d31-97ae-0ad2d1295a2c](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/0dc8260e-bd30-4247-bd4d-935e095c0956)
## VALUES OF MSE,MAE AND RMSE
![263028721-6793057c-b446-4e92-b11d-bdf3cba4a17b](https://github.com/shivanshyam79/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151513860/dbbc264b-54ef-4742-8fa7-735f00891871)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
