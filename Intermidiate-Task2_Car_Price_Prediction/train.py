# %%
# ---- Import required libraries ----

import pandas as pd            # For loading and manipulating tabular data (DataFrames)
import numpy as np             # For numerical operations (arrays, math functions)
import matplotlib.pyplot as plt  # For creating plots/charts
import seaborn as sns          # Built on top of matplotlib, makes prettier statistical plots

# Setting a consistent visual style for all our seaborn plots
sns.set_style("whitegrid")

# %%
# ---- Load the dataset ----

# read_csv() reads a CSV file and converts it into a Pandas DataFrame
# A DataFrame is like an Excel sheet in Python — rows and columns, with labels
df = pd.read_csv("data/car_data.csv", sep="\t")

# Display the first 5 rows to sanity-check that loading worked correctly
df.head()
# %%
# ---- Basic structure of the dataset ----

print("Shape of dataset (rows, columns):", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types of each column:")
print(df.dtypes)

# %%
# ---- Statistical summary of numeric columns ----
print(df.describe())
# %%
# ---- Check for missing values ----
print(df.isnull().sum())
# %%
# ---- Categorical column distributions ----
categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
for col in categorical_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())

# %%
# ---- Visualization 1: Distribution of Selling Price ----
plt.figure(figsize=(8, 5))
sns.histplot(df["Selling_Price"], bins=30, kde=True)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price (Lakhs)")
plt.savefig("selling_price_distribution.png")  # saves the plot as an image file
plt.show()

# %%
# ---- Visualization 2: Correlation heatmap (numeric features only) ----
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])  # only numeric columns
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# %%
# ---- Visualization 3: Selling Price vs Present Price (scatter) ----
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Present_Price", y="Selling_Price", hue="Fuel_Type", data=df)
plt.title("Selling Price vs Present Price")
plt.savefig("price_vs_price.png")
plt.show()

# %%
# ---- Visualization 4: Boxplot - Selling Price by Transmission ----
plt.figure(figsize=(6, 5))
sns.boxplot(x="Transmission", y="Selling_Price", data=df)
plt.title("Selling Price by Transmission Type")
plt.savefig("price_by_transmission.png")
plt.show()

# %%
# ---- Visualization 5: Selling Price by Fuel Type ----
plt.figure(figsize=(6, 5))
sns.boxplot(x="Fuel_Type", y="Selling_Price", data=df)
plt.title("Selling Price by Fuel Type")
plt.savefig("price_by_fuel.png")
plt.show()

# %%
# ---- 4.1 Drop duplicate rows ----
print("Before dropping duplicates:", df.shape)
df = df.drop_duplicates()
print("After dropping duplicates:", df.shape)

# %%
# ---- 4.2 Handle the Kms_Driven outlier ----
# We identified a bike with 500,000 km — physically implausible, likely a data entry error.
# Let's look at the top 5 highest Kms_Driven values to confirm.
print(df.sort_values("Kms_Driven", ascending=False)[["Car_Name", "Kms_Driven"]].head())

# We'll remove rows where Kms_Driven is unrealistically high (beyond 300,000 km)
df = df[df["Kms_Driven"] < 300000]
print("After removing Kms_Driven outlier:", df.shape)

# %%
# ---- 4.3 Feature Engineering: Create 'Car_Age' instead of raw 'Year' ----
# Why: A raw year like "2014" has no inherent meaning to a model.
# What actually matters for price is HOW OLD the car is, not the calendar year.
# Since this dataset is from ~2018-2019 (based on max Year=2018), we calculate age from 2019.
current_year = 2019
df["Car_Age"] = current_year - df["Year"]
df = df.drop("Year", axis=1)  # drop original Year column, we don't need it anymore

# %%
# ---- 4.4 Drop 'Car_Name' column ----
# Why: Car_Name has too many unique values (98 distinct car models) — if we one-hot encode this,
# it creates 98 new columns, causing the "curse of dimensionality" and overfitting risk,
# especially since we only have ~300 rows total.
# For a first version of this model, we drop it. (Advanced improvement: group into brand categories later.)
print("Unique car names:", df["Car_Name"].nunique())
df = df.drop("Car_Name", axis=1)

# %%
# ---- 4.5 Encode categorical columns ----
# ML models only understand numbers, not text like "Petrol" or "Manual".
# We use one-hot encoding: each category becomes its own 0/1 column.
df = pd.get_dummies(df, columns=["Fuel_Type", "Seller_Type", "Transmission"], drop_first=True)

# drop_first=True avoids the "dummy variable trap" — e.g., for Transmission (Manual/Automatic),
# we only need ONE column (Transmission_Manual: 1 or 0) since knowing it's not Manual means it's Automatic.
# This avoids redundant, perfectly correlated columns which can confuse some models.

print(df.head())
print("\nFinal columns:", df.columns.tolist())
print("Final shape:", df.shape)