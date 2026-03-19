'''
25 feb 2026
wednesday(MA'am code)

Naive Bayse Algorithm 
types -
GaussianNB, MultinomialNB and BernaulliNB
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix

#load dataset
data = pd.read_csv("insurance_data.csv")

#display first few rows
data.head()

X=data[['age']] #input feature
y=data['bought_insurance'] #output label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=48
)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Create Navie Byes model
model=BernoulliNB()

#train the model 
model.fit(X_train,y_train)

#predict on test data
y_pred =model.predict(X_test)

#accuracy
accuracy = accuracy_score(y_test,y_pred)
print("Model Accuracy: ",accuracy)

cm=confusion_matrix(y_test,y_pred)
print(cm)

#generate age value for plotting
age_range=np.linspace(X.min(),X.max(), 300).reshape(-1,1)
#reshape(-1,1) converts the array into a 2D column vector

#predict probabilities
probabilities = model.predict_proba(age_range)[:,1]
#select all rows and second column

#plot 
plt.figure()
plt.scatter(X,y)
plt.plot(age_range,probabilities)
plt.xlabel("Age")
plt.ylabel("Probability fo buying Insurance")
plt.title("Naive Bayse")
plt.show()