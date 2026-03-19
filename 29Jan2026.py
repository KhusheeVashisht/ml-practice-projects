'''
Docstring for 29Jan2026

Exp 2.2
Implementing Decision Tree using Wine Dataset (Ma'am Code)
- Creating a model
- Training a model
- Calculate Gini Index (GI) and Gain
- Design a Decision Tree
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# Load Wine Dataset
# ===============================
wine = load_wine()

data = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

data["target"] = wine.target

# ===============================
# Feature & Target
# ===============================
X = data.drop("target", axis=1)
y = data["target"]

# y contains:
# 0 -> Class_0
# 1 -> Class_1
# 2 -> Class_2

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Gini Index Function
# ===============================
def gini_index(labels):
    classes, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

print("Gini Index of Target Variable:", gini_index(y))

# ===============================
# Information Gain using Gini
# ===============================
def information_gain_gini(parent, left_child, right_child):
    w_left = len(left_child) / len(parent)
    w_right = len(right_child) / len(parent)

    return gini_index(parent) - (
        w_left * gini_index(left_child) +
        w_right * gini_index(right_child)
    )

# Example split using a Wine feature (alcohol)
threshold = X["flavanoids"].median()

left = y[X["flavanoids"] <= threshold]
right = y[X["flavanoids"] > threshold]

print("Information Gain (Gini) for Alcohol feature:",
      information_gain_gini(y, left, right))

# ===============================
# Decision Tree Model
# ===============================
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# Train the model
dt_model.fit(X_train, y_train)

# ===============================
# Model Evaluation
# ===============================
y_pred = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n",
      classification_report(
          y_test, y_pred,
          target_names=wine.target_names
      ))

print("\nConfusion Matrix:\n",
      confusion_matrix(y_test, y_pred))

# ===============================
# Plot Decision Tree
# ===============================
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    max_depth=3,   # limit depth for clarity
    fontsize=8
)
plt.tight_layout()
plt.show()
