import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Ontario_Health.csv'  # Update this to your dataset path
df = pd.read_csv(file_path)

# Step 1: Dataset Overview
print("Dataset Information:")
print(df.info())  # Check data types and non-null counts
print("\nFirst 5 rows of the dataset:")
print(df.head())  # Preview the dataset
print("\nSummary statistics:")
print(df.describe(include='all'))  # Summary statistics for numerical and categorical columns

# Step 2: Missing Values
print("\nMissing Values:")
print(df.isnull().sum())  # Count of missing values for each column
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

# Step 3: Duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Step 4: Categorical Columns Analysis
categorical_columns = df.select_dtypes(include=['object']).columns
print("\nCategorical Columns Analysis:")
for col in categorical_columns:
    print(f"Column: {col}")
    print(f"Unique values: {df[col].nunique()}")
    print(f"Top values:\n{df[col].value_counts().head()}")
    print()

# Visualize distributions of categorical data
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, y=col, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.show()

# Step 5: Numerical Columns Analysis
numerical_columns = df.select_dtypes(include=[np.number]).columns
print("\nNumerical Columns Analysis:")
for col in numerical_columns:
    print(f"Column: {col}")
    print(f"Mean: {df[col].mean()}, Median: {df[col].median()}, Std: {df[col].std()}")
    print(f"Min: {df[col].min()}, Max: {df[col].max()}")
    print()

# Visualize distributions of numerical data
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# Step 6: Correlation Matrix
if len(numerical_columns) > 1:
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

# Step 7: General Insights
print("\nGeneral Insights:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Number of categorical columns: {len(categorical_columns)}")
print(f"Number of numerical columns: {len(numerical_columns)}")

print("\nSummary:")
print("This script provides an overview of your dataset, including missing values, duplicates, distributions, and correlations.")