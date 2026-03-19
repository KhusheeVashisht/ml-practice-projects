'''
Docstring for 28Jan2026
Ma'am code in class

Exp 2.2
implementing decision tree using vizon 
-creating a model
-training a model
-calculate Gini Index (GI) and gain
-design a decision tree 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix



# Load Iris Dataset
iris = load_iris()

data = pd.DataFrame(
    iris.data, columns=iris.feature_names
)

data["target"] = iris.target
data

X = data.drop("target", axis=1)

y = data["target"]
#y contains : 0 -> setosa , 1-> vericolor , 2 -> virginica

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2,random_state=42
    )
def gini_index(labels):
    classes, counts = np.unique(labels, return_counts= True)
    probabilities = counts/ counts.sum()
    gini = 1- np.sum(probabilities ** 2)
    return gini

print("Gini Index of Target Variable :", gini_index(y))

def information_gain_gini(parent,left_child, right_child):
    weight_left=len(left_child)/len(parent)
    weight_right=len(right_child)/len(parent)

    gain = gini_index(parent) - ( weight_left * gini_index(left_child) + weight_right *             gini_index(right_child))
    return gain

'''
Exmaple split (petal lenght feature)
'''
threshold = X["sepal length (cm)"].median()

left= y[X["sepal length (cm)"] <= threshold]
right = y[X["sepal length (cm)"]> threshold]
print("Information Gain (Gini) : ", information_gain_gini(y,left,right)
)
# creating decision tree model
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)
#training the model
dt_model.fit(X_train,y_train)
#model Evaluation
y_pred = dt_model.predict(X_test)

print("Accuracy :",accuracy_score(y_test,y_pred))
print("\nClassification Report :\n",classification_report(y_test,y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))

plt.figure(figsize=(28, 14))
plot_tree(
    dt_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    max_depth=3,   # 👈 KEY FIX
    fontsize=10    # 👈 improves readability
)
plt.tight_layout()
plt.show()
