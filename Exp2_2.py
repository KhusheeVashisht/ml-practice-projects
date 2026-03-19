'''
Docstring for 28Jan2026
My code by using the iris.csv not loading the iris dataset from sklearn 

Exp 2.2
Implementing Decision Tree using CSV Iris dataset
- Creating a model
- Training a model
- Calculate Gini Index (GI) and Gain
- Design a Decision Tree
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Iris Dataset (CSV)
data = pd.read_csv("iris.csv")

# Encode target column
data["target"] = data["species"].map({
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
})

# Features and target
X = data.drop(["species", "target"], axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Gini Index
def gini_index(labels):
    classes, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

print("Gini Index of Target Variable:", gini_index(y))

# Information Gain using Gini
def information_gain_gini(parent, left_child, right_child):
    w_left = len(left_child) / len(parent)
    w_right = len(right_child) / len(parent)

    return gini_index(parent) - (
        w_left * gini_index(left_child) +
        w_right * gini_index(right_child)
    )

# Example split on petal length
threshold = X["petal_length"].median()

left = y[X["petal_length"] <= threshold]
right = y[X["petal_length"] > threshold]

print("Information Gain (Gini):", information_gain_gini(y, left, right))

# Decision Tree Model
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# Train model
dt_model.fit(X_train, y_train)

# Evaluation
y_pred = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Decision Tree
plt.figure(figsize=(28, 16))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["setosa", "versicolor", "virginica"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.show()
