
import pandas as pd

# Load the dataset
df = pd.read_csv("data/house_price.csv")

# Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
for column in df.columns:
    print(column)


print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

print("\n--- Target Price Summary ---")
print(df["Price"].describe())


print("\n--- Data Quality Report ---")
print("Missing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nDuplicate rows:", df.duplicated().sum())
