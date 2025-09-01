# Implementation-of-Linear-Regression-Using-Gradient-Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Load necessary libraries for data handling, metrics, and visualization.

2.Load Data: Read the dataset using pd.read_csv() and display basic information.

3.Initialize Parameters: Set initial values for slope (m), intercept (c), learning rate, and epochs.

4.Gradient Descent: Perform iterations to update m and c using gradient descent.

5.Plot Error: Visualize the error over iterations to monitor convergence of the model.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Praveen Raj G
RegisterNumber: 212224040245
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta_=learning_rate*(1/len(X1))*X.T.dot(errors)
        pass
    return theta
```
```
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
```
<img width="747" height="170" alt="image1" src="https://github.com/user-attachments/assets/55a24350-289f-4a46-a2cd-66e6cfc1439e" />

```
X=(data.iloc[1:, :-2].values)
print(X)
```
<img width="523" height="733" alt="image-2" src="https://github.com/user-attachments/assets/cb5914d0-7cc0-48c8-9139-8a637ac05c0f" />


```
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
```
<img width="173" height="735" alt="image-3" src="https://github.com/user-attachments/assets/b67d1a29-66ab-40a8-96bf-52517eb4a42d" />

```
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
```
<img width="552" height="757" alt="image-4" src="https://github.com/user-attachments/assets/8e1cc67f-703d-476a-a71e-9e1b94356676" />

```
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```
<img width="585" height="73" alt="img-5" src="https://github.com/user-attachments/assets/1948633d-1c1c-4723-a345-d8cc4a655be1" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
