# 26 Feb 2026
# K-Means Clustering on Student Dropout Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -------------------------------
# LOAD DATASET
# -------------------------------
data = pd.read_csv("student_dropout_behavior_dataset.csv")

print("First 5 rows:")
print(data.head())

print("\nColumns in dataset:")
print(data.columns)

# -------------------------------
# HANDLE MISSING / ZERO VALUES
# -------------------------------
numeric_cols = [
    'total_lectures', 'lectures_attended', 'total_lab_sessions',
    'labs_attended', 'quiz1_marks', 'quiz2_marks', 'quiz3_marks',
    'assignments_submitted', 'previous_gpa', 'age'
]

# Fill NaNs with mean (or 0 if mean is NaN)
for col in numeric_cols:
    col_mean = data[col].mean()
    fill_value = 0 if pd.isna(col_mean) else col_mean
    data[col].fillna(fill_value, inplace=True)

# Avoid division by zero
data['total_lectures'].replace(0, 1, inplace=True)
data['total_lab_sessions'].replace(0, 1, inplace=True)

# -------------------------------
# DERIVED FEATURES
# -------------------------------
data['AttendanceRate'] = data['lectures_attended'] / data['total_lectures']
data['LabAttendanceRate'] = data['labs_attended'] / data['total_lab_sessions']
data['PreviousGrade'] = data['previous_gpa']
data['QuizAvg'] = data[['quiz1_marks', 'quiz2_marks', 'quiz3_marks']].mean(axis=1)
data['StudyEffort'] = data['assignments_submitted'] + data['QuizAvg']

# -------------------------------
# EXPERIMENT: Academic + Behavioral Features
# -------------------------------
features = ['StudyEffort', 'AttendanceRate', 'PreviousGrade', 'LabAttendanceRate', 'age']
X = data[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_

print("\n--- K-Means Clustering (Academic + Behavioral Features) ---")
print("Inertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, kmeans.labels_))

# -------------------------------
# VISUALIZATION WITH PCA (2D)
# -------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=data['Cluster'],
    cmap='viridis',
    s=80,
    alpha=0.8,
)
plt.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    c='red',
    marker='X',
    s=200,
    edgecolor='black',
    linewidths=1.5,
)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering on Student Dataset (PCA 2D Projection)')
plt.grid(True)
plt.tight_layout()
plt.show()