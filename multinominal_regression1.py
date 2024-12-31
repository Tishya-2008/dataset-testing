import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# File path to the CSV dataset
file_path = 'dataset.csv'

# Load the dataset into a DataFrame
df = pd.read_csv(file_path)

# Print dataset info and preview
print(df.info())  # To check data types
print(df.head())  # To inspect the first few rows

# Check the column names to find the correct target column
print("Columns in the dataset:", df.columns)

# Step 1: Clean the data
# Handle missing values for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])  # Replace missing values with mode

# Handle missing values for numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())  # Replace missing values with median

# Remove duplicates
df.drop_duplicates(inplace=True)

# Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Handle categorical columns: Label Encoding
label_encoder = LabelEncoder()
for col in categorical_columns:
    if col in df.columns:  # Ensure the column exists in the DataFrame
        df[col] = label_encoder.fit_transform(df[col])

# Step 2: Preprocess the features
# Assume 'source_facility_type' is the target column (change this to your target column)
y = df['source_facility_type'].values  # Replace with the actual target column name

# Preprocess the features: Drop columns that are not relevant for model training
X = df.drop(['source_facility_type', 'index', 'facility_name'], axis=1)  # Drop irrelevant columns
X = X.select_dtypes(include=[np.number])  # Select only numeric columns

# Handle missing values (optional, depending on dataset)
X = X.fillna(X.mean())  # Fill missing values with mean of the column

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display dataset information after preprocessing
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Step 4: Initialize and train the Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model using classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Step 7: Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()