# Exp 4.1 - 20 Newsgroups Text Classification with UMAP Visualization

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# -----------------------------
# Step 1: Load dataset
# -----------------------------
categories = ['alt.atheism', 'comp.graphics', 'sci.space', 'rec.sport.hockey']  # subset for simplicity
newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# -----------------------------
# Step 2: Convert text to word counts
# -----------------------------
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

# -----------------------------
# Step 3: Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 4: Train MultinomialNB
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Predict and evaluate
# -----------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Step 6: Convert to dense once for visualization
# -----------------------------
X_dense = X_counts.toarray()

# -----------------------------
# Step 7: UMAP Visualization
# -----------------------------
reducer = umap.UMAP(n_components=2)
X_umap = reducer.fit_transform(X_dense)

plt.figure(figsize=(10,7))
sns.scatterplot(
    x=X_umap[:,0], y=X_umap[:,1],
    hue=[newsgroups.target_names[i] for i in y],
    palette='bright',
    alpha=0.7
)
plt.title("20 Newsgroups (4 categories) - UMAP 2D Projection")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend()
plt.show()
