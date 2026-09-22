import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/heart_clean.csv")

print("Dataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# 1. TARGET DISTRIBUTION
# ==========================================

plt.figure(figsize=(6, 4))

sns.countplot(x="num", data=df)

plt.title("Heart Disease Distribution")
plt.xlabel("Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

plt.show()

# ==========================================
# 2. AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="age", hue="num", bins=15, kde=True)

plt.title("Age Distribution by Heart Disease")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.show()

# ==========================================
# 3. CHOLESTEROL vs DISEASE
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(x="num", y="chol", data=df)

plt.title("Cholesterol vs Heart Disease")
plt.xlabel("Disease (0 = No, 1 = Yes)")
plt.ylabel("Cholesterol")

plt.show()

# ==========================================
# 4. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.show()