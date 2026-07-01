import pandas as pd

# Load dataset
df = pd.read_csv("data/GlobalWeatherRepository.csv")

# Basic inspection
print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)
print(df.columns)

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)
print(df.info())