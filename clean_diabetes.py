import pandas as pd

# Load dataset
df = pd.read_csv("data/diabetes.csv")

print("Original Shape:", df.shape)

# Columns where 0 should be treated as missing
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace 0 with NaN
df[zero_columns] = df[zero_columns].replace(0, pd.NA)

print("\nMissing values after replacing 0:")
print(df.isnull().sum())

# Fill missing values using median
for column in zero_columns:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Outcome"].value_counts())

# Save cleaned dataset
df.to_csv("data/diabetes_clean.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Location: data/diabetes_clean.csv")