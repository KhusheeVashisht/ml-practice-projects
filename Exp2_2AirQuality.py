'''
Docstring for 29Jan2026

Exp 2.2
Implementing Decision Tree using Air Quality CSV dataset
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


data = pd.read_csv("AirQuality.csv", sep=";")

# Drop extra unnamed column
data.drop(columns=["Unnamed: 15"], errors="ignore", inplace=True)

# Replace missing values
data.replace(-200, np.nan, inplace=True)

# Selected features
features = [
    "CO(GT)", "NO2(GT)", "NOx(GT)",
    "C6H6(GT)", "T", "RH", "AH"
]

# ===============================
# FIX: Convert string numbers to float
# ===============================
for col in features:
    data[col] = data[col].astype(str).str.replace(",", ".", regex=False)
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Drop rows with missing values
data.dropna(inplace=True)

# ===============================
# Feature & Target
# ===============================
X = data[features]

# Create target variable
def air_quality_label(co):
    if co <= 2:
        return 0      # Good
    elif co <= 5:
        return 1      # Moderate
    else:
        return 2      # Poor

data["AQ_Category"] = data["CO(GT)"].apply(air_quality_label)
y = data["AQ_Category"]

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Gini Index
# ===============================
def gini_index(labels):
    classes, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

print("Gini Index of Target Variable:", gini_index(y))

# ===============================
# Information Gain (Gini)
# ===============================
def information_gain_gini(parent, left_child, right_child):
    w_left = len(left_child) / len(parent)
    w_right = len(right_child) / len(parent)

    return gini_index(parent) - (
        w_left * gini_index(left_child) +
        w_right * gini_index(right_child)
    )

# Example split using CO(GT)
threshold = X["CO(GT)"].median()

left = y[X["CO(GT)"] <= threshold]
right = y[X["CO(GT)"] > threshold]

print("Information Gain (Gini):", information_gain_gini(y, left, right))

# ===============================
# Decision Tree Model
# ===============================
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

dt_model.fit(X_train, y_train)

# ===============================
# Model Evaluation
# ===============================
y_pred = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(
          y_test, y_pred,
          labels=[0, 1, 2],
          target_names=["Good", "Moderate", "Poor"],
          zero_division=0
      ))

print("\nConfusion Matrix:\n",
      confusion_matrix(y_test, y_pred))

# ===============================
# Plot Decision Tree
# ===============================
plt.figure(figsize=(30, 16))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["Good", "Moderate", "Poor"],
    filled=True,
    rounded=True,
    max_depth=3,   
    fontsize=9
)
plt.tight_layout()
plt.show()
