import pandas as pd

# Load dataset
df = pd.read_csv("data/breast_cancer.csv")

print("Original Shape:", df.shape)

# Convert target:
# Original: 0 = Malignant, 1 = Benign
# New:      0 = Benign, 1 = Malignant

df["Outcome"] = df["Outcome"].map({
    0: 1,
    1: 0
})

print("\nNew Target Distribution:")
print(df["Outcome"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())

# Save cleaned dataset
df.to_csv("data/breast_cancer_clean.csv", index=False)

print("\nCleaned Breast Cancer dataset saved successfully!")
print("Location: data/breast_cancer_clean.csv")