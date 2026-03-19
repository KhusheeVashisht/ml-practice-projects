import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Shopping_data.csv')

X = df[['Age']]
y = df['Spending Score (1-100']

X_train, X_test, y_train , y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

model = LinearRegression()

model.fit(X_train , y_train)

y_pred = model.predict (X_test)

mse = mean_squared_error(y_test ,  y_pred)

print (f"Mean Squared Error: {mse}")

plt.scatter(X_train, y_train , color= 'blue' , label='Training data')
plt.scatter(X_test, y_test, color='green',label='Testing data')

plt.plot(X_test,y_pred,color='red',linewidth=2,label='Regression line')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()