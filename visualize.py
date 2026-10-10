
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/house_price.csv")

# Create output folder
os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.savefig("outputs/price_distribution.png")
plt.show()
plt.close()

# 2. Area vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Area_SqFt", y="Price")
plt.title("House Area vs Price")
plt.tight_layout()
plt.savefig("outputs/area_vs_price.png")
plt.show()
plt.close()

# 3. Price by Property Type
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Property_Type", y="Price")
plt.title("Price by Property Type")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/price_by_property_type.png")
plt.show()
plt.close()

# 4. Correlation heatmap for numeric columns
plt.figure(figsize=(8, 5))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()
plt.close()

print("EDA visualizations completed!")
print("Charts saved in the outputs folder.")
