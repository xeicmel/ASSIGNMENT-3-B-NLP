import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("dataset_detecting_politeness (1).csv")

# Select required columns
data = df[["Original TikTok Comment", "Label"]].copy()

# Separate input and target
X = data["Original TikTok Comment"]
y = data["Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Display results
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

print("\nNumber of TF-IDF features:", len(vectorizer.get_feature_names_out()))

print("\nSample TF-IDF features:")
print(vectorizer.get_feature_names_out()[:20])