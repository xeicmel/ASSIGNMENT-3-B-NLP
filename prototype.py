import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("dataset_detecting_politeness (1).csv")

# Select required columns
data = df[["Original TikTok Comment", "Label"]].copy()

# Input and target
X = data["Original TikTok Comment"]
y = data["Label"]

# TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Train model using the full dataset
model = MultinomialNB()
model.fit(X_tfidf, y)

# User input
print("=== TikTok Politeness Detector ===")
comment = input("Enter a TikTok comment: ")

# Transform input
comment_tfidf = vectorizer.transform([comment])

# Prediction
prediction = model.predict(comment_tfidf)

print("\nPrediction:", prediction[0])