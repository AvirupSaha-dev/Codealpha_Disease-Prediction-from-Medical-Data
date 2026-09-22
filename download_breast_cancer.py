import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Target
df["Outcome"] = data.target

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nTarget Distribution:")
print(df["Outcome"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())

# Save dataset
df.to_csv("data/breast_cancer.csv", index=False)

print("\nBreast Cancer dataset saved successfully!")
print("Location: data/breast_cancer.csv")