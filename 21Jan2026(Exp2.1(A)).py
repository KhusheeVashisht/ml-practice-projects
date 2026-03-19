import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load Breast Cancer Dataset
data = pd.read_csv("breast-cancer.csv")   # make sure file name is correct

# Encode Diagnosis: M = 1, B = 0
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Select independent and dependent variables
X = data[['radius_mean', 'texture_mean']]
y = data['diagnosis']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Prediction for a new patient
sample_input = np.array([[14.5, 20.0]])  # radius_mean, texture_mean
sample_scaled = scaler.transform(sample_input)

prediction = model.predict(sample_scaled)
probability = model.predict_proba(sample_scaled)

print("Prediction (0 = Benign, 1 = Malignant):", prediction)
print("Probability:", probability)

# ---------- SIGMOID CURVE VISUALIZATION ----------

# Vary radius_mean, keep texture_mean constant
radius_range = np.linspace(
    X['radius_mean'].min(),
    X['radius_mean'].max(),
    300
).reshape(-1, 1)

texture_mean_const = X['texture_mean'].mean()
texture_column = np.full((300, 1), texture_mean_const)

X_plot = np.hstack([radius_range, texture_column])
X_plot_scaled = scaler.transform(X_plot)

# Manual sigmoid computation
z = np.dot(X_plot_scaled, model.coef_.T) + model.intercept_
sigmoid = 1 / (1 + np.exp(-z))

plt.figure()
plt.scatter(X['radius_mean'], y, alpha=0.4, label="Actual data")
plt.plot(radius_range.ravel(), sigmoid.ravel(), linewidth=2, label="Sigmoid curve")
plt.xlabel("Radius Mean")
plt.ylabel("Probability of Malignant")
plt.title("Sigmoid Curve for Breast Cancer Prediction")
plt.legend()
plt.show()

