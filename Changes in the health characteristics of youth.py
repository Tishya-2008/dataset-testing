import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and clean the dataset
data = pd.read_csv('Changes in the health characteristics of youth.csv')

# Clean the data
data.columns = data.columns.str.strip()  # Remove extra spaces in column names
data['Reported indicator in 2019'] = data['Reported indicator in 2019'].replace({',': ''}, regex=True).astype(float)
data['Reported indicator in 2023'] = data['Reported indicator in 2023'].replace({',': ''}, regex=True).astype(float)

# Create a column for the difference (this can be used for trend analysis)
data['difference'] = data['Reported indicator in 2023'] - data['Reported indicator in 2019']

# Prepare for linear regression: Using the years 2019 and 2023 to predict 2025 and 2026
years = np.array([2019, 2023]).reshape(-1, 1)
indicators = data[['Reported indicator in 2019', 'Reported indicator in 2023']].mean(axis=0).values  # Average for each indicator

# Fit a linear regression model to predict future years
model = LinearRegression()
model.fit(years, indicators)

# Predict values for 2025 and 2026
future_years = np.array([2025, 2026]).reshape(-1, 1)
predictions = model.predict(future_years)

# Plot the graph
plt.figure(figsize=(10, 6))

# Plot the existing data (2019 and 2023)
plt.plot([2019, 2023], indicators, label='Reported Indicator (2019, 2023)', marker='o', color='blue')

# Plot the predicted values (2025, 2026)
plt.plot([2025, 2026], predictions, label='Predicted Indicator (2025, 2026)', marker='o', color='red')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Reported Indicator')
plt.title('Predicted Changes in Health Characteristics of Youth (2019-2026)')
plt.legend()

# Show the graph
plt.grid(True)
plt.show()

# Print predictions
print(f"Predicted values for 2025: {predictions[0]}")
print(f"Predicted values for 2026: {predictions[1]}")