import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load your dataset (replace 'your_dataset.csv' with your actual file name or path)
data = pd.read_csv('USDA_yield_county.csv')

# Assuming 'Year' and 'Country' are features, and 'Yield' is the target variable
features = data[['Year', 'Country']]  # Add other relevant features
target = data['Value']

# Create a new DataFrame for storing predicted yields for each country
predicted_data_lr = pd.DataFrame(columns=['Country', 'Predicted_Yield_LR'])
predicted_data_sarima = pd.DataFrame(columns=['Country', 'Predicted_Yield_SARIMA'])

# Create a dictionary to store historical yields for each country
historical_yields = {}

# Iterate over unique countries
for country in data['Country'].unique():
    # Filter data for the current country
    country_data = data[data['Country'] == country]

    # Check the number of data points for the current country
    if len(country_data) < 2:  # Adjust as needed based on your requirements
        print(f'Skipping {country} due to insufficient data points.')
        continue

    # Split the data into training and testing sets for Linear Regression
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(country_data[['Year']],
                                                                    country_data['Value'], test_size=0.2, random_state=42)

    # Create a linear regression model
    model_lr = LinearRegression()

    # Train the linear regression model
    model_lr.fit(X_train_lr, y_train_lr)

    # Make predictions on the test set for Linear Regression
    predictions_lr = model_lr.predict(X_test_lr)

    # Split the data into training and testing sets for SARIMA
    X_train_sarima, X_test_sarima, y_train_sarima, y_test_sarima = train_test_split(country_data[['Year']],
                                                                                    country_data['Value'],
                                                                                    test_size=0.2, random_state=42)

    # Create a SARIMA model
    order = (1, 1, 1)  # (p, d, q) - Tune these parameters based on your data
    seasonal_order = (1, 1, 1, 12)  # (P, D, Q, S) - Tune these parameters based on your data
    model_sarima = SARIMAX(y_train_sarima, order=order, seasonal_order=seasonal_order)

    # Train the SARIMA model
    model_fit_sarima = model_sarima.fit(disp=False)

    # Make predictions on the test set for SARIMA
    forecast_sarima = model_fit_sarima.get_forecast(steps=len(X_test_sarima))
    predictions_sarima = forecast_sarima.predicted_mean.values

    # Store historical yields for the country
    historical_yields[country] = country_data['Value'].tolist()

    # Make predictions for 2024 for Linear Regression
    new_data_lr = pd.DataFrame({'Year': [2024]})
    predicted_yield_lr = model_lr.predict(new_data_lr[['Year']])

    # Make predictions for 2024 for SARIMA
    forecast_sarima_2024 = model_fit_sarima.get_forecast(steps=1)
    predicted_yield_sarima = forecast_sarima_2024.predicted_mean.values[0]

    # Store predicted yields for both models
    predicted_data_lr = predicted_data_lr.append({'Country': country, 'Predicted_Yield_LR': predicted_yield_lr[0]},
                                                 ignore_index=True)
    predicted_data_sarima = predicted_data_sarima.append({'Country': country, 'Predicted_Yield_SARIMA': predicted_yield_sarima},
                                                         ignore_index=True)

# Combine the predictions from both models into a single DataFrame
final_predictions = pd.merge(predicted_data_lr, predicted_data_sarima, on='Country')

# Plot final predicted yields for each country in a single bar graph with rotated x-axis labels
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

bar_width = 0.4
bar_positions_lr = np.arange(len(final_predictions))
bar_positions_sarima = np.arange(len(final_predictions)) + bar_width

# Plot predicted yields for 2024 (Linear Regression)
plt.bar(bar_positions_lr, final_predictions['Predicted_Yield_LR'].values,
        width=bar_width, color='blue', label='Linear Regression', alpha=0.7)

# Plot predicted yields for 2024 (SARIMA)
plt.bar(bar_positions_sarima, final_predictions['Predicted_Yield_SARIMA'].values,
        width=bar_width, color='red', label='SARIMA', alpha=0.7)

# Add labels and legend with rotated x-axis labels
plt.xticks(np.arange(len(final_predictions)) + bar_width/2, final_predictions['Country'], rotation=45, ha='right')
plt.title('Final Predicted Yields Comparison - Linear Regression vs SARIMA')
plt.xlabel('Country')
plt.ylabel('Predicted Yield')
plt.legend()
plt.tight_layout()  # Ensure tight layout to prevent label clipping
plt.show()
