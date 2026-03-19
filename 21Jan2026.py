import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("insurance_data.csv")

data.head()
X= data[['age']]
y=data['bought_insurance']

X_train, X_test , y_train , y_test =train_test_split(X,y,test_size=0.25,random_state=40)

model = LogisticRegression()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Model Accuracy:",accuracy)

age=[[35]]
prediction=model.predict(age)
probability=model.predict_proba(age)
print("Prediction (0=no,1=yes):",prediction)
print("Probability :",probability)

age_range=np.linspace(X['age'].min(),X['age'].max(),300).reshape(-1,1)# coverts the array into 2D column vector 
probabilities=model.predict_proba(age_range)[:,1]

plt.figure()
plt.scatter(X['age'],y)
plt.plot(age_range,probabilities)
plt.xlabel("Age")
plt.ylabel("Probability of buying Insurance")
plt.title("Logistic Regression S-Curve (Sigmoid)")
plt.show()