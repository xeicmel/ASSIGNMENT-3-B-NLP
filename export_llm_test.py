
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset_detecting_politeness (1).csv")

# Select required columns
data = df[["Original TikTok Comment", "Label"]].copy()

# Separate input and target
X = data["Original TikTok Comment"]
y = data["Label"]

# Recreate the same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create test dataset
llm_test = pd.DataFrame({
    "Comment": X_test,
    "True_Label": y_test
})

# Reset index
llm_test = llm_test.reset_index(drop=True)

# Export test dataset
llm_test.to_csv("llm_test_dataset.csv", index=False)

# Export comments only for LLM input
llm_input = llm_test[["Comment"]]
llm_input.to_csv("llm_input.csv", index=False)

print("LLM test dataset exported successfully!")
print("Total test comments:", len(llm_test))
print("\nFiles created:")
print("- llm_test_dataset.csv (comments and true labels)")
print("- llm_input.csv (comments only)")
