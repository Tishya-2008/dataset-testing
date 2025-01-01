import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the dataset
file_path = 'Emergency Visits for Patients Residing in the LGA.csv'
df = pd.read_csv(file_path)

# Step 1: Rename Columns for Clarity
df.columns = [
    "Year",
    "Location",
    "CTAS Level",
    "Population",
    "Emergency Visit Rate (per 1,000)",
    "Notes"
]

# Step 2: Prepare Target Variable
# Here, we'll bin the "Emergency Visit Rate (per 1,000)" into categories for multinomial regression
# Adjust the binning as per your use case
df["Visit Rate Category"] = pd.cut(
    pd.to_numeric(df["Emergency Visit Rate (per 1,000)"], errors='coerce'),
    bins=[0, 50, 100, 150, np.inf],
    labels=["Low", "Medium", "High", "Very High"]
)

# Drop rows with missing target values
df.dropna(subset=["Visit Rate Category"], inplace=True)

# Step 3: Prepare Features and Target
X = df.drop(columns=["Emergency Visit Rate (per 1,000)", "Visit Rate Category", "Notes"])  # Features
y = df["Visit Rate Category"]  # Target

# Encode categorical variables for features
X = pd.get_dummies(X, drop_first=True)

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train the Multinomial Logistic Regression Model
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Step 8: Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Step 9: Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the model: {accuracy:.2f}")