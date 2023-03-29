# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
```


## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: D.vishnu vardhan reddy
RegisterNumber:  212221230023
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
#splitting train and test data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#displaying the predicted values
y_pred
y_test
#graph plot for traing data
plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,regressor.predict(x_train),color='orange')
plt.title("Hours vs Scores(training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
![image](https://user-images.githubusercontent.com/94175324/228479646-436135b0-7a31-46c4-a160-baa86cb54fa1.png)
![image](https://user-images.githubusercontent.com/94175324/228479789-25a27de5-1ea9-4649-beb3-b4ac15ffb314.png)
![image](https://user-images.githubusercontent.com/94175324/228480053-51b45ec5-cc2b-414a-a9cc-7dd5093f3ce1.png)
![image](https://user-images.githubusercontent.com/94175324/228480168-7ec8e48f-3d38-4f10-87e5-96e274a9238b.png)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
