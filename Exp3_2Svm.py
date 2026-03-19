'''
Docstring for Exp3.2(Svm)
18feb,2026
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
# 2. Preprocessing
# -------------------------------
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

# Convert target to numeric
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# -------------------------------
# 3. Select ONLY 2 Features for 2D Plot
# -------------------------------
X = data[["tenure", "MonthlyCharges"]]
y = data["Churn"]

# -------------------------------
# 4. Train-Test Split
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
# 6. SVM Model (Changed Kernel)
# -------------------------------
svm_model = SVC(kernel='poly', degree=3, C=1.0)
svm_model.fit(X_train, y_train)

# -------------------------------
# 7. Decision Boundary Plot
# -------------------------------
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train,
    cmap=plt.cm.coolwarm,
    marker='o',
    label="Train Data"
)

plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_test,
    cmap=plt.cm.coolwarm,
    marker='x',
    label="Test Data"
)

plt.xlabel("Tenure (Standardized)")
plt.ylabel("Monthly Charges (Standardized)")
plt.title("SVM Decision Boundary (Polynomial Kernel)")
plt.legend()
plt.show()

# -------------------------------
# 8. Evaluation
# -------------------------------
y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
