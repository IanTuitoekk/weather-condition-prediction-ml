import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create folders if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load dataset
df = pd.read_csv("data/GlobalWeatherRepository.csv")

print("=" * 50)
print("INITIAL DATASET SHAPE")
print("=" * 50)
print(df.shape)

# --------------------------------------------------
# Missing Values
# --------------------------------------------------
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing_values = df.isnull().sum()
print(missing_values)

# Missing values visualization
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values Visualization")
plt.tight_layout()
plt.savefig("reports/missing_values.png")
plt.close()

# --------------------------------------------------
# Duplicate Records
# --------------------------------------------------
duplicates = df.duplicated().sum()

print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)
print("Duplicate rows:", duplicates)

df.drop_duplicates(inplace=True)

# --------------------------------------------------
# Data Types
# --------------------------------------------------
print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

# --------------------------------------------------
# Outlier Detection
# --------------------------------------------------
print("\n" + "=" * 50)
print("OUTLIER ANALYSIS")
print("=" * 50)

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Generate boxplots for first three numerical columns
for col in numeric_cols[:3]:

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(f"reports/boxplot_{col}.png")
    plt.close()

# IQR Method
for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"{col}: {len(outliers)} outliers")

# --------------------------------------------------
# Save Cleaned Dataset
# --------------------------------------------------
df.to_csv("data/GlobalWeatherRepository_cleaned.csv", index=False)

print("\nCleaning completed successfully.")
print("Cleaned dataset saved as:")
print("data/GlobalWeatherRepository_cleaned.csv")