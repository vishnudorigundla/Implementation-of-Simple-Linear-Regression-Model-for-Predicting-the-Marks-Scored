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
1. df.head()


![image](https://user-images.githubusercontent.com/94175324/229284622-06351aad-a34c-4d95-b4aa-4f1db414c55e.png)


2. df.tail()


![image](https://user-images.githubusercontent.com/94175324/229284637-0e0a1dcd-5db6-4b91-bad7-966698e47096.png)


3. Array value of X


![image](https://user-images.githubusercontent.com/94175324/229284651-e1bee598-c81e-4777-8f91-ea141e057c0a.png)


4. Array value of Y


![image](https://user-images.githubusercontent.com/94175324/229284659-c1e13e67-31bd-45d1-9461-7cee0c865204.png)


5. Values of Y prediction


![image](https://user-images.githubusercontent.com/94175324/229284670-7236b3b0-8a60-4d66-a6ca-05ffaca9c1ca.png)


6. Array values of Y test


![image](https://user-images.githubusercontent.com/94175324/229284673-04c8fa53-a549-42d5-9a7e-38b426cab262.png)


7. Training Set Graph


![image](https://user-images.githubusercontent.com/94175324/229284682-91580971-3de2-40c5-8940-1b89c341ba16.png)




8. Values of MSE, MAE and RMSE


![image](https://user-images.githubusercontent.com/94175324/229284731-3721097e-8610-4d4b-bf82-7126caabb568.png)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
