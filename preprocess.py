
import pandas as pd
import os

# Load the original dataset
df = pd.read_csv("data/house_price.csv")

print("Missing values before preprocessing:")
print(df.isnull().sum())

# Fill missing numeric values with the median
df["Area_SqFt"] = df["Area_SqFt"].fillna(
    df["Area_SqFt"].median()
)

df["Rooms"] = df["Rooms"].fillna(
    df["Rooms"].median()
)

# Fill missing categorical values with the most frequent category
df["Furnishing"] = df["Furnishing"].fillna(
    df["Furnishing"].mode()[0]
)

# Create output folder
os.makedirs("data/processed", exist_ok=True)

# Save the cleaned dataset
df.to_csv("data/processed/house_price_cleaned.csv", index=False)

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

print("\nCleaned dataset shape:", df.shape)
print("Cleaned dataset saved successfully!")
