'''
Docstring for 14Jan2026
This is a program for Multiple Linear Regression using
Independent Variables: Age and Annual Income.
The model predicts Spending Score and is evaluated using MSE.
'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('Shopping_data.csv')

# Independent variables
X = df[['Age', 'Annual Income (k$)']]

# Dependent variable (FIXED column name)
y = df['Spending Score (1-100']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualization (Actual vs Predicted)
plt.scatter(y_test, y_pred, color='purple', alpha=0.7)

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='black',
    linestyle='--',
    label='Perfect Prediction'
)

plt.xlabel('Actual Spending Score')
plt.ylabel('Predicted Spending Score')
plt.title('Actual vs Predicted Spending Score')
plt.legend()
plt.show()
