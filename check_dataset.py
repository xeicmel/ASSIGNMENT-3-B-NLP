import pandas as pd

# Load dataset
df = pd.read_csv("dataset_detecting_politeness (1).csv")

# Display dataset information
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLabel Distribution:")
print(df["Label"].value_counts())