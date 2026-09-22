import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/breast_cancer_clean.csv")

print("Dataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nTarget Distribution:")
print(df["Outcome"].value_counts())

# 1. Cancer distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Outcome", data=df)
plt.title("Breast Cancer Distribution")
plt.xlabel("Outcome (0 = Benign, 1 = Malignant)")
plt.ylabel("Number of Patients")
plt.show()

# 2. Mean radius distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="mean radius", hue="Outcome", bins=20, kde=True)
plt.title("Mean Radius Distribution")
plt.xlabel("Mean Radius")
plt.ylabel("Number of Patients")
plt.show()

# 3. Mean texture vs cancer
plt.figure(figsize=(8, 5))
sns.boxplot(x="Outcome", y="mean texture", data=df)
plt.title("Mean Texture vs Breast Cancer")
plt.xlabel("Outcome (0 = Benign, 1 = Malignant)")
plt.ylabel("Mean Texture")
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Breast Cancer Feature Correlation")
plt.show()