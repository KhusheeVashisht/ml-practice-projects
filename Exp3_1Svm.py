'''
Docstring for Exp3.1(Svm)
18 feb,2026
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("Telco_Cusomer_Churn.csv")

# -------------------------------
# 2. Data Preprocessing
# -------------------------------

# Drop customerID (not useful for prediction)
data.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Fill missing values
data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

# Convert categorical columns to numeric
data = pd.get_dummies(data, drop_first=True)

# -------------------------------
# 3. Define Features and Target
# -------------------------------
X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]

# -------------------------------
# 4. Split Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=48
)

# -------------------------------
# 5. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 6. Create & Train SVM Model
# -------------------------------
svm_model = SVC(kernel='rbf', C=1.0, random_state=48)
svm_model.fit(X_train, y_train)

# -------------------------------
# 7. Make Predictions
# -------------------------------
y_pred = svm_model.predict(X_test)

# -------------------------------
# 8. Evaluate Model
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
