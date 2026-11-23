import pandas as pd

# Load dataset
df = pd.read_csv("dataset_detecting_politeness (1).csv")

# Select the columns needed for classification
data = df[["Original TikTok Comment", "Label"]].copy()

# Check duplicate comments
print("Duplicate comments:", data["Original TikTok Comment"].duplicated().sum())

# Check labels
print("\nUnique labels:")
print(data["Label"].unique())

# Display selected data
print("\nSelected Data:")
print(data.head())

print("\nDataset size:")
print(data.shape)