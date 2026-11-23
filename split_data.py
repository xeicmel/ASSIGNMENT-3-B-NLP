import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset_detecting_politeness (1).csv")

# Select required columns
data = df[["Original TikTok Comment", "Label"]].copy()

# Split input and target
X = data["Original TikTok Comment"]
y = data["Label"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display dataset sizes
print("Training data:", len(X_train))
print("Testing data:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())