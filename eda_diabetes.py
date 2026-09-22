import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/diabetes_clean.csv")

print("Dataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nTarget Distribution:")
print(df["Outcome"].value_counts())

# 1. Diabetes distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Number of Patients")
plt.show()

# 2. Glucose distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Glucose", hue="Outcome", bins=20, kde=True)
plt.title("Glucose Distribution by Diabetes")
plt.xlabel("Glucose")
plt.ylabel("Number of Patients")
plt.show()

# 3. BMI vs Diabetes
plt.figure(figsize=(8, 5))
sns.boxplot(x="Outcome", y="BMI", data=df)
plt.title("BMI vs Diabetes")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("BMI")
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Diabetes Feature Correlation")
plt.show()