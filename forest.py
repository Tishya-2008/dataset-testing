import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Given dataset
data = {
    "Indicator": [
        "Perceived health, excellent or very good",
        "Perceived mental health, excellent or very good",
        "Life satisfaction, satisfied or very satisfied",
        "Met sleep guidelines in the last 7 days",
        "Perceived life stress, most days quite a bit or extremely stressful"
    ],
    "2019": [
        423500, 705000, 332900, 508800, 214400
    ],
    "2023": [
        210500, 190400, 114100, 391300, 444500
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create an empty list to store the results
results = []

# Prepare the years and values
years = np.array([2019, 2023]).reshape(-1, 1)
future_years = np.array([2025, 2026]).reshape(-1, 1)

# Loop through each indicator and fit a linear regression
for i, indicator in enumerate(df["Indicator"]):
    # Get the actual values for 2019 and 2023
    y_values = np.array([df["2019"][i], df["2023"][i]])

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(years, y_values)

    # Predict for 2025 and 2026
    predictions = model.predict(future_years)

    # Calculate R^2 and coefficients
    r2 = r2_score(y_values, model.predict(years))
    coef = model.coef_[0]  # Since we have one predictor (year)

    # Store the results for each indicator
    results.append({
        "Indicator": indicator,
        "Coefficient": coef,
        "R^2": r2,
        "Prediction_2025": predictions[0],
        "Prediction_2026": predictions[1]
    })

    # Plotting the data and regression line
    plt.figure(figsize=(8, 6))
    plt.scatter([2019, 2023], y_values, color='blue', label='Actual Data')
    plt.plot([2019, 2023, 2025, 2026], 
             np.concatenate([y_values, predictions]), 
             color='red', linestyle='--', label='Fitted Line')

    plt.title(f'Regression for {indicator}')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.xticks([2019, 2023, 2025, 2026])
    plt.legend()
    plt.grid(True)
    plt.show()

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results)
print(results_df)