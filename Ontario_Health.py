import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

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

# Handle missing values (example: fill with mode for categorical and median for numerical columns)
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

numerical_columns = df.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

# Step 3: Duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Step 4: Categorical Columns Analysis
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

# Step 8: Confusion Matrix Analysis
# Define target and features
target_column = 'Access Level'  # Replace with your actual target column
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces in column names
y = df[target_column]
X = df.drop(target_column, axis=1)

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Encode the target variable (if needed)
y = pd.factorize(y)[0]  # Converts to numerical labels

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()