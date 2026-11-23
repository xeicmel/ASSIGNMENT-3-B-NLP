
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("dataset_detecting_politeness (1).csv")

# Select required columns
data = df[["Original TikTok Comment", "Label"]].copy()

X = data["Original TikTok Comment"]
y = data["Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define models
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Store results
results = []

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Weighted)": precision_score(y_test, y_pred, average="weighted"),
        "Recall (Weighted)": recall_score(y_test, y_pred, average="weighted"),
        "F1-score (Weighted)": f1_score(y_test, y_pred, average="weighted")
    })

# Display comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))
