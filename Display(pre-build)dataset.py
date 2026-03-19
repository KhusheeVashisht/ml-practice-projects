'''
25 feb 2026
'''
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Load a few categories for simplicity
categories = ['alt.atheism', 'comp.graphics', 'sci.space', 'rec.sport.hockey']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# Convert to DataFrame
df = pd.DataFrame({
    "Category": [newsgroups.target_names[i] for i in newsgroups.target],
    "Text": newsgroups.data
})

df_sample = pd.DataFrame({
    "Category": [newsgroups.target_names[i] for i in newsgroups.target[:50]],
    "Text Snippet": [text[:200]+"..." for text in newsgroups.data[:50]]
})
df_sample.to_csv("20newsgroups_sample_snippet.csv", index=False)

# Save first 50 rows only (for easier screenshot)
df.head(50).to_csv("20newsgroups_sample.csv", index=False)

# Optional: Show first 10 rows
print(df.head(50))