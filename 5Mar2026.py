import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("breast-cancer.csv")

# Drop unnecessary column
data = data.drop(["id"], axis=1)

# Encode diagnosis
le = LabelEncoder()
data["diagnosis"] = le.fit_transform(data["diagnosis"])

# Split features and label
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Logistic Regression
# -----------------------
lr = LogisticRegression(max_iter=5000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, pred_lr)
cm_lr = confusion_matrix(y_test, pred_lr)

print("Logistic Regression")
print(classification_report(y_test, pred_lr))

# -----------------------
# SVM
# -----------------------
svm = SVC()
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

acc_svm = accuracy_score(y_test, pred_svm)
cm_svm = confusion_matrix(y_test, pred_svm)

print("SVM")
print(classification_report(y_test, pred_svm))

# -----------------------
# Decision Tree
# -----------------------
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

acc_dt = accuracy_score(y_test, pred_dt)
cm_dt = confusion_matrix(y_test, pred_dt)

print("Decision Tree")
print(classification_report(y_test, pred_dt))

# ===============================
# Plot Confusion Matrices
# ===============================

fig, ax = plt.subplots(1,3, figsize=(15,4))

sns.heatmap(cm_lr, annot=True, fmt="d", ax=ax[0])
ax[0].set_title("Logistic Regression")

sns.heatmap(cm_svm, annot=True, fmt="d", ax=ax[1])
ax[1].set_title("SVM")

sns.heatmap(cm_dt, annot=True, fmt="d", ax=ax[2])
ax[2].set_title("Decision Tree")

plt.show()

# ===============================
# Accuracy Comparison Plot
# ===============================

models = ["Logistic Regression","SVM","Decision Tree"]
accuracy = [acc_lr, acc_svm, acc_dt]

plt.figure(figsize=(6,4))
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# ===============================
# Decision Tree Visualization
# ===============================

plt.figure(figsize=(15,8))
plot_tree(dt, filled=True, feature_names=X.columns)
plt.title("Decision Tree Structure")
plt.show()